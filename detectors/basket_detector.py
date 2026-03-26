from ultralytics import YOLO
import cv2
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.device_utils import get_device

class BasketDetector:
    """YOLO-based hoop (basket) detector."""

    def __init__(self,
                 model_path: str = "weights/Basketball_v1.pt",
                 class_num: int = 1,
                 best_detection: bool = True,
                 confidence_threshold: float = 0.1,
                 imgsz: int = 416,
                 half: bool = False,
                 retina_masks: bool = False,
                 stream: bool = False,):
        self.model_path = model_path
        self.class_num = class_num
        self.best_detection = best_detection
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.half = half
        self.retina_masks = retina_masks
        self.stream = stream
        self.device = get_device()

        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def process(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on one BGR frame."""
        if frame is None or frame.size == 0:
            return self._empty_result()

        results = self.model(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            retina_masks=self.retina_masks,
            stream=self.stream,
            verbose=False
        )

        detections = self._parse_detections(results)

        return detections

    def _parse_detections(self, results) -> List[Dict]:
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                center_x, center_y = int(x1 + w/2), int(y1 + h/2)

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id != self.class_num:
                    continue

                if confidence < self.confidence_threshold:
                    continue

                detection = {
                    'class_id': class_id,
                    'class_name': "basket",
                    'confidence': confidence,
                    'bbox': (x1, y1, w, h),
                    'center': (center_x, center_y),
                    'area': w * h
                }

                detections.append(detection)

        if self.best_detection and detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            return [best_detection]

        return detections

    def _empty_result(self) -> List[Dict]:
        return []


def main():
    print("Basket detector test")

    detector = BasketDetector()

    cap = cv2.VideoCapture("inputs/test.mp4")

    print("Detecting; press 'q' to quit")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.process(frame)

        if detections:
            detection = detections[0]
            x1, y1, w, h = detection['bbox']

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        cv2.imshow('Basket Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
