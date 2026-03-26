from ultralytics import YOLO
import cv2
import numpy as np
from utils import get_device
import math
from typing import Dict, List, Tuple, Optional

class PoseDetector:
    def __init__(self, model_resolution=640, stream=True, half=True, retina_masks=True, best_detection=True):
        self.model = YOLO("weights/Pose.pt")
        self.stream = stream
        self.model_resolution = model_resolution
        self.device = get_device()
        self.half = half
        self.retina_masks = retina_masks
        self.best_detection = best_detection

        self.keypoint_colors = {
            'pose': (0, 255, 0),
            'face': (0, 0, 255),
            'left_arm': (255, 0, 0),
            'right_arm': (255, 165, 0),
            'left_leg': (255, 255, 0),
            'right_leg': (255, 0, 255),
        }

        self.skeleton = [
            [5, 6], [5, 11], [6, 12], [11, 12],
            [5, 7], [7, 9],
            [6, 8], [8, 10],
            [11, 13], [13, 15],
            [12, 14], [14, 16],
            [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [0, 6]
        ]

        self.keypoints = []

        self.frame_count = 0
        self.pose_history = []

        self.shoe_bottom_points = []

    def process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """Detect pose on one frame and draw skeleton."""
        try:
            if frame is None:
                print("Warning: Received None frame in PoseDetector")
                return np.zeros((480, 640, 3), dtype=np.uint8)

            self.frame = frame.copy()
            self.frame_count = frame_count

            self.shoe_bottom_points = []

            try:
                results = self.model(self.frame, stream=self.stream, device=self.device,
                                   imgsz=self.model_resolution, half=self.half, retina_masks=self.retina_masks)

                for r in results:
                    if r.keypoints is not None:
                        self.keypoints = []

                        try:
                            keypoints = r.keypoints.data
                            boxes = r.boxes

                            person_indices_to_process = []
                            if self.best_detection and boxes is not None and len(boxes) > 0 and len(keypoints) > 0:
                                person_confidences = []
                                min_count = min(len(boxes), len(keypoints))
                                for box_idx in range(min_count):
                                    try:
                                        confidence = float(boxes[box_idx].conf[0])
                                        person_confidences.append((box_idx, confidence))
                                    except Exception:
                                        continue

                                if person_confidences:
                                    best_person = max(person_confidences, key=lambda x: x[1])
                                    person_indices_to_process = [best_person[0]]
                                else:
                                    person_indices_to_process = [0] if len(keypoints) > 0 else []
                            else:
                                person_indices_to_process = list(range(len(keypoints)))

                            for person_idx in person_indices_to_process:
                                if person_idx >= len(keypoints):
                                    continue

                                try:
                                    kpts = keypoints[person_idx]

                                    person_keypoints = []
                                    for kpt in kpts:
                                        try:
                                            x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
                                            if conf > 0.5:
                                                person_keypoints.append((int(x), int(y), conf))
                                            else:
                                                person_keypoints.append(None)
                                        except Exception as kpt_error:
                                            print(f"Error processing keypoint: {kpt_error}")
                                            person_keypoints.append(None)

                                    self.keypoints.append(person_keypoints)

                                    try:
                                        self.draw_skeleton(person_keypoints)
                                    except Exception as draw_error:
                                        print(f"Error drawing skeleton: {draw_error}")

                                except Exception as person_error:
                                    print(f"Error processing person {person_idx}: {person_error}")
                                    continue

                        except Exception as keypoints_error:
                            print(f"Error processing keypoints: {keypoints_error}")

                if len(self.keypoints) > 0:
                    self.pose_history.append((self.keypoints, self.frame_count))
                    if len(self.pose_history) > 30:
                        self.pose_history.pop(0)

            except Exception as model_error:
                print(f"Error in pose detection model: {model_error}")
                return self.frame

            return self.frame

        except Exception as e:
            print(f"Critical error in PoseDetector process_frame: {e}")
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def draw_point(self, point: Tuple[int, int], color, size=5):
        x, y = point
        cv2.circle(self.frame, (x, y), size, color, -1)
        return (x, y)

    def draw_line(self, point1: Tuple[int, int], point2: Tuple[int, int], color, thickness=2):
        cv2.line(self.frame, point1, point2, color, thickness)

    def draw_skeleton(self, keypoints: List):
        if keypoints is None or len(keypoints) == 0:
            return

        shoe_height = 36
        person_shoe_bottom_points = []

        for i, kpt in enumerate(keypoints):
            if kpt is not None:
                x, y, conf = kpt
                x, y = int(x), int(y)
                if i <= 4:
                    color = self.keypoint_colors['face']
                elif i in [5, 6, 11, 12]:
                    color = self.keypoint_colors['pose']
                elif i in [7, 9]:
                    color = self.keypoint_colors['left_arm']
                elif i in [8, 10]:
                    color = self.keypoint_colors['right_arm']
                elif i in [13, 15]:
                    color = self.keypoint_colors['left_leg']
                elif i in [14, 16]:
                    color = self.keypoint_colors['right_leg']
                else:
                    color = (0, 255, 0)

                self.draw_point((x, y), color)

                if i == 15 or i == 16:
                    self.process_foot_extensions(i, keypoints, (x, y), shoe_height, person_shoe_bottom_points)

        if person_shoe_bottom_points:
            self.shoe_bottom_points.append(person_shoe_bottom_points)

        self.draw_skeleton_lines(keypoints)

    def process_foot_extensions(self, foot_idx: int, keypoints: List, foot_point: Tuple[int, int],
                               shoe_height: int, person_shoe_bottom_points: List):
        x, y = foot_point
        foot_color = self.keypoint_colors['left_leg'] if foot_idx == 15 else self.keypoint_colors['right_leg']

        shoe_bottom_x = x
        shoe_bottom_y = y + shoe_height

        self.draw_point((shoe_bottom_x, shoe_bottom_y), foot_color)
        self.draw_line((x, y), (shoe_bottom_x, shoe_bottom_y), foot_color)

        foot_index = 0 if foot_idx == 15 else 1
        person_shoe_bottom_points.append((foot_index, (shoe_bottom_x, shoe_bottom_y)))

        knee_idx = 13 if foot_idx == 15 else 14

        if keypoints[knee_idx] is not None:
            knee_x, knee_y, _ = keypoints[knee_idx]
            knee_x, knee_y = int(knee_x), int(knee_y)
            extended_point = self.calculate_extended_point((knee_x, knee_y), (x, y), shoe_height)

            self.draw_point(extended_point, foot_color)
            self.draw_line((x, y), extended_point, foot_color)

            foot_index = 2 if foot_idx == 15 else 3
            person_shoe_bottom_points.append((foot_index, extended_point))

    def draw_skeleton_lines(self, keypoints: List):
        for pair in self.skeleton:
            idx1, idx2 = pair
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                if keypoints[idx1] is not None and keypoints[idx2] is not None:
                    x1, y1, _ = keypoints[idx1]
                    x2, y2, _ = keypoints[idx2]
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))

                    if idx1 <= 4 and idx2 <= 6:
                        color = self.keypoint_colors['face']
                    elif (idx1 in [5, 6, 11, 12] and idx2 in [5, 6, 11, 12]):
                        color = self.keypoint_colors['pose']
                    elif (idx1 in [5, 7] and idx2 in [7, 9]) or (idx2 in [5, 7] and idx1 in [7, 9]):
                        color = self.keypoint_colors['left_arm']
                    elif (idx1 in [6, 8] and idx2 in [8, 10]) or (idx2 in [6, 8] and idx1 in [8, 10]):
                        color = self.keypoint_colors['right_arm']
                    elif (idx1 in [11, 13] and idx2 in [13, 15]) or (idx2 in [11, 13] and idx1 in [13, 15]):
                        color = self.keypoint_colors['left_leg']
                    elif (idx1 in [12, 14] and idx2 in [14, 16]) or (idx2 in [12, 14] and idx1 in [14, 16]):
                        color = self.keypoint_colors['right_leg']
                    else:
                        color = (0, 255, 0)

                    self.draw_line(pt1, pt2, color)

    def calculate_extended_point(self, start_point: Tuple[int, int], end_point: Tuple[int, int], distance: int) -> Tuple[int, int]:
        """Extend from end_point along the direction start_point -> end_point by distance (pixels)."""
        dir_x = end_point[0] - start_point[0]
        dir_y = end_point[1] - start_point[1]

        length = math.sqrt(dir_x**2 + dir_y**2)
        if length > 0:
            dir_x = (dir_x / length) * distance
            dir_y = (dir_y / length) * distance

            extended_x = int(end_point[0] + dir_x)
            extended_y = int(end_point[1] + dir_y)
            return (extended_x, extended_y)

        return end_point

    def get_head_center(self) -> Optional[Tuple[int, int]]:
        """Mean of visible face keypoints (indices 0-4), or None."""
        if len(self.keypoints) == 0:
            return None

        person_keypoints = self.keypoints[0]
        if len(person_keypoints) < 5:
            return None

        face_keypoints = []
        for i in range(5):
            if person_keypoints[i] is not None:
                x, y, _ = person_keypoints[i]
                face_keypoints.append((x, y))

        if len(face_keypoints) == 0:
            return None

        avg_x = sum(pt[0] for pt in face_keypoints) // len(face_keypoints)
        avg_y = sum(pt[1] for pt in face_keypoints) // len(face_keypoints)

        return (avg_x, avg_y)

    def get_foot_points(self) -> List[Tuple]:
        """Foot keypoints 15-16 plus extended sole points when drawn."""
        foot_points = []

        for person_idx, person_keypoints in enumerate(self.keypoints):
            if len(person_keypoints) >= 17:
                left_foot = person_keypoints[15]
                right_foot = person_keypoints[16]

                if left_foot is not None or right_foot is not None:
                    person_foot_points = []

                    if left_foot is not None:
                        person_foot_points.append(left_foot[:2])

                    if right_foot is not None:
                        person_foot_points.append(right_foot[:2])

                    if person_idx < len(self.shoe_bottom_points):
                        for foot_index, bottom_point in self.shoe_bottom_points[person_idx]:
                            person_foot_points.append(bottom_point)

                    if person_foot_points:
                        foot_points.append(person_foot_points)

        return foot_points
