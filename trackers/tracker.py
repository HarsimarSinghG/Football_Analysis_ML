# Importing necessary libraries
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # YOLO model for object detection
import supervision as sv  # Supervision library for tracking
import pickle  # For saving and loading serialized data
import os  # For file and directory operations
import sys  # For manipulating the Python runtime environment
import pandas as pd  # For data manipulation and analysis

# Adding the parent directory to the system path for module imports
sys.path.append("../")

# Helper function to calculate the center of a bounding box
def get_center_of_bbox(bbox):
    """
    Calculate the center point (x, y) of a bounding box.
    :param bbox: List [x_min, y_min, x_max, y_max] representing the bounding box.
    :return: Tuple (x_center, y_center).
    """
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

# Helper function to calculate the width of a bounding box
def get_width_of_bbox(bbox):
    """
    Calculate the width of a bounding box.
    :param bbox: List [x_min, y_min, x_max, y_max] representing the bounding box.
    :return: Integer width of the bounding box.
    """
    return bbox[2] - bbox[0]

# Extracts player tracking information from detections
def return_player_tracking_info(frame_number, tracked_detections, class_names):
    """
    Extract player tracking data from tracked detections.
    :param frame_number: Integer frame index.
    :param tracked_detections: List of detections including bounding boxes, class IDs, and track IDs.
    :param class_names: Dictionary mapping class names to their IDs.
    :return: Dictionary of player tracking data with track IDs as keys.
    """
    player_dict = {}
    for element in tracked_detections:
        bounding_box = element[0].tolist()
        class_id = element[3]
        track_id = element[4]
        if class_id == class_names["player"]:
            player_dict[track_id] = {"bounding_box": bounding_box}
    return player_dict

# Extracts referee tracking information from detections
def return_referee_tracking_info(frame_number, tracked_detections, class_names):
    """
    Extract referee tracking data from tracked detections.
    :param frame_number: Integer frame index.
    :param tracked_detections: List of detections including bounding boxes, class IDs, and track IDs.
    :param class_names: Dictionary mapping class names to their IDs.
    :return: Dictionary of referee tracking data with track IDs as keys.
    """
    referee_dict = {}
    for element in tracked_detections:
        bounding_box = element[0].tolist()
        class_id = element[3]
        track_id = element[4]
        if class_id == class_names["referee"]:
            referee_dict[track_id] = {"bounding_box": bounding_box}
    return referee_dict

# Extracts ball tracking information from detections
def return_ball_tracking_info(frame_number, tracked_detections, class_names):
    """
    Extract ball tracking data from tracked detections.
    :param frame_number: Integer frame index.
    :param tracked_detections: List of detections including bounding boxes, class IDs, and track IDs.
    :param class_names: Dictionary mapping class names to their IDs.
    :return: Dictionary of ball tracking data with a fixed track ID of 1.
    """
    ball_dict = {}
    for element in tracked_detections:
        bounding_box = element[0].tolist()
        class_id = element[3]
        if class_id == class_names["ball"]:
            # There is only one ball in the frame, so using 1 as the tracking ID.
            ball_dict[1] = {"bounding_box": bounding_box}
    return ball_dict

# Tracker class for managing detection and tracking
class Tracker:
    def __init__(self, model_path):
        """
        Initialize the Tracker with a YOLO model and ByteTrack tracker.
        :param model_path: Path to the pre-trained YOLO model file.
        """
        self.model = YOLO(model_path)  # Load YOLO model
        self.tracker = sv.ByteTrack()  # Initialize ByteTrack tracker

    def return_all_tracking_info(self, frames):
        """
        Generate tracking information for players, referees, and the ball across multiple frames.
        :param frames: List of frames (images) to process.
        :return: Dictionary containing tracking information for players, referees, and the ball.
        """
        detections = self.prediction_in_frames(frames)  # Get detections for all frames
        info = {"players": [], "referees": [], "ball": []}  # Initialize result dictionary
        for i in range(len(detections)):
            ith_detection = detections[i]
            all_classes = ith_detection.names  # Get class name mappings
            reversed_class_names = {v: k for k, v in all_classes.items()}  # Reverse mapping for easy lookup
            supervision_detections = sv.Detections.from_ultralytics(ith_detection)  # Convert detections to Supervision format
            tracking_detection = self.tracker.update_with_detections(supervision_detections)  # Update tracker
            # Append tracking info for players, referees, and ball
            info["players"].append(return_player_tracking_info(i, tracking_detection, reversed_class_names))
            info["referees"].append(return_referee_tracking_info(i, tracking_detection, reversed_class_names))
            info["ball"].append(return_ball_tracking_info(i, tracking_detection, reversed_class_names))
        return info

    def prediction_in_frames(self, frames):
        """
        Predict detections in the given frames.
        :param frames: List of frames (images) to process.
        :return: List of detection results for each frame.
        """
        # Predict in batches to improve performance
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Run model prediction with a minimum confidence of 10%
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections
