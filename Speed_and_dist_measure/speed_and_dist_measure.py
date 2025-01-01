import cv2
import numpy as np
import math


def get_center_of_bbox(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def measure_distance(point1, point2):
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

    def store_player_positions(self, tracked_objects):
        """
        Maps players' bounding box positions in video frames to real-world coordinates.

        Args:
            tracked_objects (dict): A dictionary containing tracking data for players,
                                    including their bounding boxes.
        """
        for frame_index, frame_players in enumerate(tracked_objects["players"]):
            # Loop through each frame and the players detected in that frame
            for player_key, player_info in frame_players.items():
                # Extract the bounding box for the player
                bbox = player_info["bounding_box"]

                # Calculate the center coordinates of the bounding box
                center_x, center_y = get_center_of_bbox(bbox)

                # Convert the center coordinates to a format suitable for transformation
                frame_coords = np.array([[[center_x, center_y]]], dtype='float32')

                # Apply perspective transformation to get real-world coordinates
                world_coords = cv2.perspectiveTransform(frame_coords, self.matrix)

                # Scale the transformed coordinates to match the real-world dimensions
                world_x = world_coords[0][0][0] * self.x_scale
                world_y = world_coords[0][0][1] * self.y_scale

                # Store the real-world coordinates in the tracked objects data
                tracked_objects["players"][frame_index][player_key]["real_world_position"] = (world_x, world_y)

    def compute_distance_and_velocity(self, frame_data, tracked_objects):
        """
        Calculates the distance traveled and velocity for players over time.

        Args:
            frame_data (list): Frames of the video data.
            tracked_objects (dict): A dictionary containing tracking data for players,
                                    including their real-world positions.
        """
        # Dictionary to store cumulative distances traveled by each player
        cumulative_distances = {}

        # Extract player data from the tracked objects
        player_data = tracked_objects["players"]
        total_frames = len(tracked_objects)

        # Process frames in steps defined by the step interval
        for start_frame in range(0, total_frames, self.frame_window):
            # Determine the ending frame for the current interval
            end_frame = min(start_frame + self.frame_window, total_frames - 1)

            # Iterate through each player in the starting frame
            for player_id, _ in player_data[start_frame].items():
                # Skip players not present in the ending frame
                if player_id not in player_data[end_frame]:
                    continue

                # Retrieve the starting and ending positions of the player
                initial_position = player_data[start_frame][player_id]["real_world_position"]
                final_position = player_data[end_frame][player_id]["real_world_position"]

                # Skip if either position is not available
                if not initial_position or not final_position:
                    continue

                # Calculate the distance covered between the two positions
                covered_distance = measure_distance(initial_position, final_position)

                # Calculate the time elapsed between the two frames
                elapsed_time = (end_frame - start_frame) / self.frame_rate

                # Convert the distance and time to velocity in km/h
                speed_kmh = (covered_distance / elapsed_time) * 3.6

                # Store the velocity in the tracked objects data
                player_data[start_frame][player_id]["speed"] = speed_kmh

                # Update the cumulative distance traveled by the player
                cumulative_distances[player_id] = cumulative_distances.get(player_id, 0) + covered_distance

                # Store the updated cumulative distance
                player_data[start_frame][player_id]["total_distance"] = cumulative_distances[player_id]
