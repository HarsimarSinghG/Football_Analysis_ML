import cv2
import numpy as np

from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from Ball_interpolation.interpolator import Interpolator
from Annotator.annotator import Annotator
from Speed_and_dist_measure.speed_and_dist_measure import SpeedDistMeasure


def read_video_as_frames(video_path):
    """
    Return a list of frames in the video
    :param video_path: directory for the video
    :return: [Frames]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def create_video_from_frames(frames, video_path, fps=24):
    """
    Creates and saves a video from a list of frames.

    :param frames: List of frames to be saved in the video.
    :param video_path: Path where the output video will be saved.
    :param fps: Frames per second for the video.
    """
    if not frames:
        raise ValueError("No frames provided to create the video.")

    # Get the dimensions of the first frame
    frame_height, frame_width = frames[0].shape[:2]

    # Define the codec and create VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, codec, fps, (frame_width, frame_height))

    # Write each frame to the video
    for frame in frames:
        video_writer.write(frame)

    # Release the video writer object
    video_writer.release()


def main():
    # Load video frames and initialize required components
    video_frames = read_video_as_frames('dataset_video/08fd33_4.mp4')
    # Use the best.pt model that was trained in the previous step of the project
    model_path = 'models/best.pt'
    video_tracker = Tracker(model_path)
    interpolator = Interpolator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    annotator = Annotator()

    # Track objects and interpolate ball positions
    object_tracks = video_tracker.return_all_tracking_info(video_frames)
    object_tracks["ball"] = interpolator.interpolate_ball_positions(object_tracks["ball"])

    # Assign team colors to players
    team_assigner.assign_team_color(video_frames[0], object_tracks["players"][0])

    # Assign teams and team colors to players for each frame
    for frame_num, player_track in enumerate(object_tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bounding_box"], player_id)
            object_tracks["players"][frame_num][player_id]["team"] = team
            object_tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Assign ball possession to players
    team_ball_control = []
    for frame_num, player_track in enumerate(object_tracks["players"]):
        ball_bounding_box = object_tracks["ball"][frame_num][1]["bounding_box"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bounding_box)

        if assigned_player != -1:
            object_tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(object_tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else None)

    team_ball_control = np.array(team_ball_control)

    speed_dist_measure = SpeedDistMeasure()

    speed_dist_measure.save_positions(object_tracks)
    speed_dist_measure.calculate_distance_and_speed(video_frames, object_tracks)

    # Annotate frames with team and ball control information
    output_video_frames = annotator.draw_annotations(video_frames, object_tracks, team_ball_control)

    # Save the annotated frames to a new video
    create_video_from_frames(output_video_frames, 'output_video/output_video.avi')


if __name__ == '__main__':
    main()
