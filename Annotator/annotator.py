import cv2
import numpy as np


class Annotator:
    def __init__(self):
        self.rectangle_width = 40
        self.rectangle_height = 20
        self.alpha = 0.4

    def _get_center_and_width(self, bbox):
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        width = bbox[2] - bbox[0]
        return (x_center, y_center), width

    def _draw_ellipse_for_object(self, frame, bbox, color, track_id=None):
        (x_center, y_center), width = self._get_center_and_width(bbox)
        y2 = int(bbox[3])
        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)), angle=0.0, startAngle=-45,
                    endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

        if track_id is not None:
            self._draw_rectangle_with_id(frame, x_center, y2, track_id, color)

        return frame

    def _draw_rectangle_with_id(self, frame, x_center, y2, track_id, color):
        x1_rect = x_center - self.rectangle_width // 2
        x2_rect = x_center + self.rectangle_width // 2
        y1_rect = y2 - self.rectangle_height // 2 + 15
        y2_rect = y2 + self.rectangle_height // 2 + 15
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

        x1_text = x2_rect + 12
        if track_id > 99:
            x1_text -= 10
        cv2.putText(frame, str(track_id), (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_triangle_for_object(self, frame, bbox, color):
        (x_center, y_center), _ = self._get_center_and_width(bbox)
        triangle_points = np.array(
            [[x_center, y_center], [x_center - 10, y_center - 20], [x_center + 10, y_center - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def _draw_team_ball_control_overlay(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num]
        team_1_num_frames = team_ball_control_till_frame.count(team_ball_control_till_frame == 1).shape[0]
        team_2_num_frames = team_ball_control_till_frame.count(team_ball_control_till_frame == 2).shape[0]

        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames) * 100
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames) * 100

        cv2.putText(frame, f"Team 1: {team_1:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {team_2:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame

    def draw_speed_and_distance(self, frame, player_positions, frame_num):
        for player_id, position in player_positions[frame_num].items():
            distance = player_positions[frame_num][player_id]["total_distance"]
            x_center, y_center = player_positions[frame_num][player_id]["real_world_position"]
            speed = player_positions[frame_num][player_id]["speed"]
            cv2.putText(frame, f"Player {player_id} Speed: {speed:.2f} m/s", (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Player {player_id} Distance: {distance:.2f} m", (int(x_center), int(y_center) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            self._draw_object_annotations(frame, tracks["players"][frame_num], tracks["referees"][frame_num],
                                          tracks["ball"][frame_num], frame_num, team_ball_control)
            output_video_frames.append(frame)
        return output_video_frames

    def _draw_object_annotations(self, frame, player_dict, referee_dict, ball_dict, frame_num, team_ball_control):
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self._draw_ellipse_for_object(frame, player["bounding_box"], color, track_id)
            if player.get("has_ball", False):
                frame = self._draw_triangle_for_object(frame, player["bounding_box"], (0, 0, 255))
            frame = self.draw_speed_and_distance(frame, player_dict, frame_num)

        for track_id, referee in referee_dict.items():
            frame = self._draw_ellipse_for_object(frame, referee["bounding_box"], (0, 255, 255), track_id)

        for track_id, ball in ball_dict.items():
            frame = self._draw_triangle_for_object(frame, ball["bounding_box"], (0, 255, 0))

        frame = self._draw_team_ball_control_overlay(frame, frame_num, team_ball_control)
        return frame
