import cv2
import numpy as np
import math


def get_center_of_bbox(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def measure_distance(self, point1, point2):
    """
    Calculate the Euclidean distance between two points
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


class SpeedDistMeasure:
    def __init__(self):
        self.field_width, self.field_height = 100, 60
        self.image_width, self.image_height = 800, 300
        self.court_width = 68
        self.court_length = 23.32
        self.pixel_vertices = np.array([[110, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])
        self.target_vertices = np.array([[0, self.court_width],
                                         [0, 0],
                                         [self.court_length, 0],
                                         [self.court_length, self.court_width]])
        self.x_scale = self.field_width / self.image_width
        self.y_scale = self.field_height / self.image_height
        self.matrix = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
        self.player_positions = {}
        self.player_distances = {}
        self.frame_rate = 24
        self.frame_window = 5

    def save_positions(self, object_tracks):
        for frame_num, players in enumerate(object_tracks["players"]):
            for player_id, player_data in players.items():
                bounding_box = player_data["bounding_box"]
                x_center, y_center = get_center_of_bbox(bounding_box)
                position = np.array([[[x_center, y_center]]], dtype='float32')
                transformed_position = cv2.perspectiveTransform(position, self.matrix)
                real_world_x = transformed_position[0][0][0] * self.x_scale
                real_world_y = transformed_position[0][0][1] * self.y_scale
                object_tracks["players"][frame_num][player_id]["real_world_position"] = (real_world_x, real_world_y)

    def calculate_distance_and_speed(self, video_frames, object_tracks):
        total_distance = {}

        player_dict = object_tracks["players"]
        number_of_frames = len(object_tracks)
        for frame_num in range(0, number_of_frames, self.frame_window):
            last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

            for track_id, _ in player_dict[frame_num].items():
                if track_id not in player_dict[last_frame]:
                    continue

                start_position = player_dict[frame_num][track_id]["real_world_position"]
                end_position = player_dict[last_frame][track_id]["real_world_position"]

                if start_position is None or end_position is None:
                    continue

                distance_covered = measure_distance(start_position, end_position)
                time_elapsed = (last_frame - frame_num) / self.frame_rate
                speed_km_hour = (distance_covered / time_elapsed) * 3.6

                object_tracks["players"][frame_num][track_id]["speed"] = speed_km_hour

                total_distance[track_id] = total_distance.get(track_id, 0) + distance_covered

                object_tracks["players"][frame_num][track_id]["total_distance"] = total_distance[track_id]
