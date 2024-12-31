import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import pandas as pd

sys.path.append("../")

import numpy as np


def get_center_of_bbox(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def get_width_of_bbox(bbox):
    return bbox[2] - bbox[0]


def return_player_tracking_info(frame_number, tracked_detections, class_names):
    player_dict = {}
    for element in tracked_detections:
        bounding_box = element[0].tolist()
        class_id = element[3]
        track_id = element[4]
        if class_id == class_names["player"]:
            player_dict[track_id] = {"bounding_box": bounding_box}
    return player_dict


def return_referee_tracking_info(frame_number, tracked_detections, class_names):
    referee_dict = {}
    for element in tracked_detections:
        bounding_box = element[0].tolist()
        class_id = element[3]
        track_id = element[4]
        if class_id == class_names["referee"]:
            referee_dict[track_id] = {"bounding_box": bounding_box}
    return referee_dict


def return_ball_tracking_info(frame_number, tracked_detections, class_names):
    ball_dict = {}
    for element in tracked_detections:
        bounding_box = element[0].tolist()
        class_id = element[3]
        if class_id == class_names["ball"]:
            # There is only one ball in the frame so using 1 as the tracking id
            ball_dict[1] = {"bounding_box": bounding_box}
    return ball_dict


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def return_all_tracking_info(self, frames):
        detections = self.prediction_in_frames(frames)
        info = {"players": [], "referees": [], "ball": []}
        for i in range(len(detections)):
            ith_detection = detections[i]
            all_classes = ith_detection.names
            reversed_class_names = {v: k for k, v in all_classes.items()}
            supervision_detections = sv.Detections.from_ultralytics(ith_detection)
            tracking_detection = self.tracker.update_with_detections(supervision_detections)
            info["players"].append(return_player_tracking_info(i, tracking_detection, reversed_class_names))
            info["referees"].append(return_referee_tracking_info(i, tracking_detection, reversed_class_names))
            info["ball"].append(return_ball_tracking_info(i, tracking_detection, reversed_class_names))
        return info

    def prediction_in_frames(self, frames):
        """
        Predicts the detections in the given frames.
        :param frames: list of frames
        :return:
        """
        # Predicting in batches so that performance is better
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Detection is considered with at least 10% confidence
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections
