from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import score, detect_down, detect_up, clean_hoop_pos, clean_ball_pos, get_device, ThreePointMarker
from typing import Dict, Optional, List

try:
    from .ball_detector import BallDetector
    from .basket_detector import BasketDetector
except ImportError:
    from ball_detector import BallDetector
    from basket_detector import BasketDetector

class ShotDetector:
    def __init__(self, model_resolution=640, stream=False, half=False, retina_masks=False):
        self.overlay_text = "Waiting..."

        self.ball_detector = BallDetector(
            model_path="weights/Basketball_v1.pt",
            class_num=0,
            best_detection=True,
            confidence_threshold=0.3,
            imgsz=model_resolution,
            half=half,
            retina_masks=retina_masks,
            stream=stream
        )
        
        self.basket_detector = BasketDetector(
            model_path="weights/Basketball_v1.pt",
            class_num=1,
            best_detection=True,
            confidence_threshold=0.5,
            imgsz=model_resolution,
            half=half,
            retina_masks=retina_masks,
            stream=stream
        )

        self.stream = stream
        self.model_resolution = model_resolution
        self.device = get_device()
        self.half = half
        self.retina_masks = retina_masks

        self.ball_pos = []
        self.hoop_pos = []
        self.frame_count = 0
        self.frame = None
        self.pending_ball_pos = []

        self.fixed_hoop_position = None
        self.hoop_calibration_frames = []
        self.hoop_calibration_complete = False

        self.makes = 0
        self.attempts = 0

        self.shot_in_progress = False
        self.shot_ball_positions = []
        self.shot_frames = []

        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.three_point_marker = ThreePointMarker()
        self.is_three_point_mode = False

        self.is_first_shot = True
        self.retry_count = 0
        
    def start_three_point_marking(self, frame: np.ndarray):
        """Enter three-point arc marking mode."""
        self.is_three_point_mode = True
        return self.three_point_marker.start_marking(frame)
    
    def stop_three_point_marking(self):
        """Leave three-point marking mode."""
        self.is_three_point_mode = False
    
    def load_three_point_config(self, video_name: str) -> bool:
        """Load saved three-point arc from disk."""
        return self.three_point_marker.load_config(video_name)
    
    def save_three_point_config(self, video_name: str) -> str:
        """Persist three-point arc config."""
        return self.three_point_marker.save_config(video_name)
    
    def three_point_mouse_callback(self, event, x, y, flags, param):
        if self.is_three_point_mode:
            self.three_point_marker.mouse_callback(event, x, y, flags, param)
            if not self.three_point_marker.is_marking and self.is_three_point_mode:
                self.is_three_point_mode = False

    def process_frame(self, frame: np.ndarray, frame_count: int, pose_detector=None) -> np.ndarray:
        """Detect ball/hoop, update tracks; optional pose_detector skips ball near head."""
        self.frame = frame.copy()
        self.frame_count = frame_count

        if self.shot_in_progress:
            self.shot_frames.append(frame.copy())

        if self.is_three_point_mode:
            return self.three_point_marker.process_frame(self.frame)

        ball_detections = self.ball_detector.process(self.frame)

        if ball_detections:
            detection = ball_detections[0]
            center = detection['center']
            bbox = detection['bbox']
            conf = detection['confidence']

            should_skip = False
            if pose_detector is not None:
                head_center = pose_detector.get_head_center()
                if head_center is not None:
                    ball_center_x, ball_center_y = center
                    head_center_x, head_center_y = head_center
                    distance = math.sqrt((ball_center_x - head_center_x)**2 + (ball_center_y - head_center_y)**2)

                    ball_diagonal = math.sqrt(bbox[2]**2 + bbox[3]**2)
                    distance_threshold = ball_diagonal * 0.8

                    if distance < distance_threshold:
                        should_skip = True

            if should_skip:
                x1, y1, w, h = bbox
                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(0, 255, 0))
            else:
                new_point = (center, self.frame_count, bbox[2], bbox[3], conf)
                self.pending_ball_pos.append(new_point)

            if len(self.pending_ball_pos) >= 2:
                point_b = self.pending_ball_pos[0]
                point_c = self.pending_ball_pos[1]

                if len(self.ball_pos) > 0 and self._should_skip_middle_point(point_b, point_c):
                    self.pending_ball_pos.pop(0)
                else:
                    self.ball_pos.append(point_b)
                    self.pending_ball_pos.pop(0)

            x1, y1, w, h = bbox
            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(0, 255, 0))

        if self.fixed_hoop_position is not None:
            center = self.fixed_hoop_position['center']
            bbox = self.fixed_hoop_position['bbox']
            conf = self.fixed_hoop_position['confidence']

            self.hoop_pos.append((center, self.frame_count, bbox[2], bbox[3], conf))

            x1, y1, w, h = bbox
            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 0, 255))
        else:
            basket_detections = self.basket_detector.process(self.frame)
            if basket_detections:
                detection = basket_detections[0]
                center = detection['center']
                bbox = detection['bbox']
                conf = detection['confidence']

                self.hoop_pos.append((center, self.frame_count, bbox[2], bbox[3], conf))

                x1, y1, w, h = bbox
                cvzone.cornerRect(self.frame, (x1, y1, w, h), colorC=(255, 0, 0))

        self.clean_motion()

        if not self.is_three_point_mode and len(self.three_point_marker.arc_points) >= 3:
            return self.three_point_marker.draw_markers(self.frame, show_points_and_lines=False)
            
        return self.frame

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def display_score(self):
        """Score text is composed in DetectorManager."""
        pass

        if self.fade_counter > 0:
            self.fade_counter -= 1

    def reset_shot_status(self):
        """Reset layup / retry state."""
        self.is_first_shot = True
        self.retry_count = 0
        self.shot_in_progress = False
        self.shot_ball_positions = []
        self.shot_frames = []
    
    def start_hoop_calibration(self):
        """Begin averaging hoop bbox over frames."""
        self.hoop_calibration_frames = []
        self.hoop_calibration_complete = False
        self.fixed_hoop_position = None
        print("Starting hoop position calibration...")
    
    def calibrate_hoop_position(self, frame: np.ndarray) -> bool:
        if self.hoop_calibration_complete:
            return True

        basket_detections = self.basket_detector.process(frame)

        if basket_detections:
            detection = basket_detections[0]
            center = detection['center']
            bbox = detection['bbox']
            conf = detection['confidence']
            
            self.hoop_calibration_frames.append({
                'center': center,
                'bbox': bbox,
                'confidence': conf
            })
            
            print(f"Hoop calibration frame {len(self.hoop_calibration_frames)}: center={center}, conf={conf:.3f}")
            
            if len(self.hoop_calibration_frames) >= 3:
                avg_center_x = sum(d['center'][0] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)
                avg_center_y = sum(d['center'][1] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)

                avg_bbox_w = sum(d['bbox'][2] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)
                avg_bbox_h = sum(d['bbox'][3] for d in self.hoop_calibration_frames) / len(self.hoop_calibration_frames)

                base_bbox = self.hoop_calibration_frames[0]['bbox']
                avg_bbox_x = base_bbox[0]
                avg_bbox_y = base_bbox[1]

                self.fixed_hoop_position = {
                    'center': (int(avg_center_x), int(avg_center_y)),
                    'bbox': (avg_bbox_x, avg_bbox_y, int(avg_bbox_w), int(avg_bbox_h)),
                    'confidence': 1.0
                }
                
                self.hoop_calibration_complete = True
                print(f"Hoop calibration complete: fixed position = {self.fixed_hoop_position['center']}")
                return True
        
        return False
    
    def get_fixed_hoop_position(self):
        return self.fixed_hoop_position

    def check_shot(self) -> Optional[Dict]:
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return None

        if not self.shot_in_progress:
            if detect_up(self.ball_pos, self.hoop_pos):
                ball_width = self.ball_pos[-1][2]
                ball_height = self.ball_pos[-1][3]
                hoop_width = self.hoop_pos[-1][2]
                hoop_height = self.hoop_pos[-1][3]
                
                if ball_width < hoop_width and ball_height < hoop_height:
                    self.shot_in_progress = True
                    self.shot_ball_positions = []
                    self.shot_frames = []
                    print("Shot attempt started")

        if self.shot_in_progress:
            if len(self.ball_pos) > 0:
                self.shot_ball_positions.append(self.ball_pos[-1])

            if detect_down(self.ball_pos, self.hoop_pos):
                self.shot_in_progress = False
                self.attempts += 1

                is_score = score(self.shot_ball_positions, self.hoop_pos, self.shot_frames)

                self.shot_ball_positions = []
                self.shot_frames = []

                if is_score:
                    self.makes += 1
                    result = {
                        "text": "Made Shot!" if self.is_first_shot else "Made Retry Shot!",
                        "color": (0, 255, 0),
                        "event": "first_shot_made" if self.is_first_shot else "retry_shot_made"
                    }
                    self.reset_shot_status()
                    return result
                else:
                    if self.is_first_shot:
                        self.is_first_shot = False
                        self.retry_count = 0
                        return {
                            "text": "Missed First Shot - Retry Available",
                            "color": (255, 165, 0),
                            "event": "first_shot_missed"
                        }
                    else:
                        self.retry_count += 1
                        if self.retry_count >= 2:
                            self.reset_shot_status()
                            return {
                                "text": "Failed After Two Retries - Exit Three-Point Line",
                                "color": (255, 0, 0),
                                "event": "retry_shot_missed"
                            }
                        return {
                            "text": f"Missed Retry ({self.retry_count}/2) - Try Again",
                            "color": (255, 165, 0),
                            "event": "retry_shot_missed"
                        }

        return None

    def _should_skip_middle_point(self, point_b, point_c):
        """True if pending middle point b is an outlier between last track point a and c."""
        if len(self.ball_pos) < 1:
            return False

        point_a = self.ball_pos[-1][0]
        point_b_pos = point_b[0]
        point_c_pos = point_c[0]

        dist_ac = math.sqrt((point_c_pos[0] - point_a[0]) ** 2 + (point_c_pos[1] - point_a[1]) ** 2)
        dist_ab = math.sqrt((point_b_pos[0] - point_a[0]) ** 2 + (point_b_pos[1] - point_a[1]) ** 2)
        dist_bc = math.sqrt((point_c_pos[0] - point_b_pos[0]) ** 2 + (point_c_pos[1] - point_b_pos[1]) ** 2)

        if len(self.ball_pos) >= 1:
            avg_ball_size = (self.ball_pos[-1][2] + self.ball_pos[-1][3]) / 2
            threshold_close = avg_ball_size * 1
            threshold_far = avg_ball_size * 3
        else:
            threshold_close = 30
            threshold_far = 100

        if dist_ac < threshold_close and dist_ab > threshold_far and dist_bc > threshold_far:
            return True
        
        return False


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

    print("Shot detector test (inputs/standard.mp4)")
    print("Keys: q quit, r reset stats")

    detector = ShotDetector(
        model_resolution=640,
        stream=False,
        half=False,
        retina_masks=False
    )

    video_path = "inputs/standard.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")

    frame_count = 0
    print("Running...")

    cv2.namedWindow('Shot Detection Test')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_count += 1

        processed_frame = detector.process_frame(frame, frame_count)

        shot_result = detector.check_shot()
        if shot_result:
            print(f"Frame {frame_count}: {shot_result['text']}")

        info_y = 30
        cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(processed_frame, f"Makes: {detector.makes}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(processed_frame, f"Attempts: {detector.attempts}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        info_y += 30
        success_rate = (detector.makes / detector.attempts * 100) if detector.attempts > 0 else 0
        cv2.putText(processed_frame, f"Success Rate: {success_rate:.1f}%", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        info_y += 30
        shot_status = "First Shot" if detector.is_first_shot else f"Retry ({detector.retry_count}/2)"
        cv2.putText(processed_frame, f"Status: {shot_status}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Shot Detection Test', processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit")
            break
        elif key == ord('r'):
            detector.makes = 0
            detector.attempts = 0
            detector.reset_shot_status()
            print("Stats reset")

    cap.release()
    cv2.destroyAllWindows()

    print("Summary:")
    print(f"  Makes: {detector.makes}")
    print(f"  Attempts: {detector.attempts}")
    success_rate = (detector.makes / detector.attempts * 100) if detector.attempts > 0 else 0
    print(f"  Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()