import cv2
import numpy as np
from ultralytics import YOLO
from .shot_detector import ShotDetector
from .pose_detector import PoseDetector
from datetime import datetime
import os
from typing import Union, Optional, List, Tuple, Dict
from utils.transmit_utils import MessageTransmitter
import time
from utils.inputs_utils import InputType
from utils.marker_utils import StartEndLineMarker
from utils.statemachine_utils import StateMachine, Event, State
import signal
import sys
import json

class DetectorManager:
    def __init__(self, input_source: Union[str, int], message_transmitter=None, model_resolution=1280):
        """
        Args:
            input_source: File path (str), camera index (int), or rtsp:// URL.
            message_transmitter: Optional MessageTransmitter for events.
            model_resolution: YOLO input size (default 1280).
        """
        self.input_source = input_source
        self.is_camera = isinstance(input_source, int)
        self.is_rtsp = isinstance(input_source, str) and input_source.startswith("rtsp://")

        if self.is_camera:
            self.video_name = f"camera_{input_source}"
            self.cap = cv2.VideoCapture(input_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            self.video_path = input_source
            self.video_name = os.path.basename(input_source) if not self.is_rtsp else f"rtsp_{hash(input_source) % 10000}"

            if self.is_rtsp:
                self.cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            else:
                self.cap = cv2.VideoCapture(input_source)

        self.frame_count = 0
        self.frame = None
        self.display_frame = None

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30
        print(f"Video FPS: {self.video_fps}")

        self.use_real_time_timing = self.is_camera or self.is_rtsp
        timing_mode = "real-time" if self.use_real_time_timing else "frame-based"
        print(f"Timing mode: {timing_mode}")

        self.fps = 0
        self.fps_counter = 0
        self.fps_timer = time.time()

        self.processing_times = []

        self.model_resolution = model_resolution
        self.stream = True
        self.half = True
        self.retina_masks = False
        print(f"Using YOLO model resolution: {model_resolution}")

        self.use_parameter_mode = False
        self.parameter_mode_fps = 30
        self.video_base_name = None
        self.start_end_config_path = None
        self.three_point_config_path = None

        self.gender = 'M'
        self.exam_score = None
        self.score_config = None
        self._load_score_config()

        self.shot_detector = ShotDetector(model_resolution=self.model_resolution,
                                          stream=self.stream, half=self.half, retina_masks=self.retina_masks)
        self.pose_detector = PoseDetector(model_resolution=self.model_resolution // 2,
                                          stream=self.stream, half=self.half, retina_masks=self.retina_masks,
                                          best_detection=True)

        self.hoop_calibration_mode = False
        self.hoop_calibration_frames_processed = 0

        self.message_transmitter = message_transmitter

        self.setup_output()

        self.fade_frames = 2
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.overlay_text = "Waiting..."

        self.paused = False
        self.marking_three_point = False
        self.marking_start_end_lines = False

        self.warmup_mode = False

        self.players_in_three_point_area = {}
        self.violation_fade_counter = 0
        self.violation_text = ""

        self.start_end_line_marker = StartEndLineMarker()

        self.last_foot_positions = None
        self.line_crossing_cooldown = 0
        self.line_crossing_cooldown_frames = 60
        self.line_crossing_text = ""
        self.line_crossing_fade_counter = 0

        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.signal_handler)

        self.timing_active = False
        self.start_time = 0
        self.start_frame = 0
        self.elapsed_time = 0
        self.timing_text = ""
        self.timing_finished = False

        cv2.namedWindow('Frame')

        cv2.setMouseCallback('Frame', self.mouse_callback)

        self.load_three_point_config()
        self.load_start_end_lines_config()

        ret, first_frame = self.cap.read()
        if ret:
            self.frame = first_frame.copy()
            self.display_frame = first_frame.copy()
        else:
            print("Could not read first frame")

        self.last_process_time = time.time()

        self.state_machine = StateMachine()

        self.state_file = None

        self.initialize_new_state()

    def _rewind_video(self):
        """Seek file input to start; reopen capture if seek fails."""
        if self.is_camera or self.is_rtsp:
            return
        try:
            if self.cap is not None:
                if self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    return
        except Exception:
            pass
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.input_source)
        except Exception as reopen_error:
            print(f"Error rewinding video by reopen: {reopen_error}")
            return

    def _save_three_point_config_with_fallback(self, target_path: Optional[str], video_base_name: str) -> Optional[str]:
        """Save three-point config to target_path if set, else default naming."""
        try:
            if len(self.shot_detector.three_point_marker.arc_points) < 3:
                print("No three-point area to save")
                return None
            if target_path and isinstance(target_path, str) and len(target_path.strip()) > 0:
                parent_dir = os.path.dirname(target_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                data = {
                    "arc_points": self.shot_detector.three_point_marker.arc_points,
                    "version": "simplified"
                }
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                return target_path
            return self.shot_detector.save_three_point_config(video_base_name)
        except Exception as e:
            print(f"Error saving three-point config: {e}")
            return None

    def _save_start_end_config_with_fallback(self, target_path: Optional[str], video_base_name: str) -> Optional[str]:
        """Save start/end lines to target_path if set, else default naming."""
        try:
            if len(self.start_end_line_marker.line_points) == 0:
                print("No start-end lines to save")
                return None
            if target_path and isinstance(target_path, str) and len(target_path.strip()) > 0:
                parent_dir = os.path.dirname(target_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                data = {
                    "line_points": self.start_end_line_marker.line_points
                }
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                return target_path
            return self.start_end_line_marker.save_config(video_base_name)
        except Exception as e:
            print(f"Error saving start-end config: {e}")
            return None

    def load_three_point_config(self):
        video_base_name = os.path.splitext(self.video_name)[0]
        if self.shot_detector.load_three_point_config(video_base_name):
            print(f"Three-point line configuration loaded")
            return True
        return False
        
    def load_start_end_lines_config(self):
        video_base_name = os.path.splitext(self.video_name)[0]
        if self.start_end_line_marker.load_config(video_base_name):
            print(f"Start-end lines configuration loaded")
            return True
        return False

    def setup_output(self):
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read video frame; output init failed.")
        composed = self.compose_display_frame(test_frame)
        canvas_h, canvas_w = composed.shape[:2]

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(self.video_name)[0]
        self.output_path = f"outputs/{base_name}_{timestamp}.mp4"
        
        os.makedirs('outputs', exist_ok=True)

        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.output_path = self.output_path.replace('.mp4', '.avi')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (canvas_w, canvas_h)
        )

        if not self.out.isOpened():
            print(f"Warning: Failed to initialize video writer with {fourcc}")
            self.out = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (canvas_w, canvas_h)
            )
        
        print(f"Output video will be saved as: {self.output_path}")
        print(f"Output resolution: {canvas_w}x{canvas_h} at {fps} FPS (with interface)")

    def check_line_crossing(self):
        """Detect crossing start/end lines using highest-confidence person."""
        if len(self.start_end_line_marker.line_points) == 0:
            return

        foot_points_list = self.pose_detector.get_foot_points()

        if not foot_points_list or len(foot_points_list) == 0:
            self.last_foot_positions = None
            return

        person_foot_points = foot_points_list[0]

        if self.line_crossing_cooldown > 0:
            self.line_crossing_cooldown -= 1

        if self.last_foot_positions is None:
            self.last_foot_positions = person_foot_points
            return

        prev_foot_points = self.last_foot_positions

        for foot_idx, current_foot in enumerate(person_foot_points):
            if foot_idx >= len(prev_foot_points):
                continue

            prev_foot = prev_foot_points[foot_idx]

            for line_idx, line in enumerate(self.start_end_line_marker.line_points):
                if len(line) != 2:
                    continue

                if self.line_crossing_cooldown > 0:
                    continue

                p1, p2 = line[0], line[1]
                p3, p4 = prev_foot, current_foot

                if self.is_line_intersection(p1, p2, p3, p4):
                    self.line_crossing_text = f"Crossed line {line_idx+1}!"
                    self.line_crossing_fade_counter = self.fade_frames * 5

                    self.line_crossing_cooldown = self.line_crossing_cooldown_frames

                    if self.message_transmitter:
                        rel_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        self.message_transmitter.send_event(
                            self.video_name, 
                            rel_timestamp, 
                            'line_crossing',
                            self.frame_count
                        )
                    
                    current_time = time.time()

                    if not self.timing_active and not self.timing_finished:
                        if self.use_real_time_timing:
                            self.start_time = current_time
                        else:
                            self.start_frame = self.frame_count
                            self.start_time = current_time

                        self.timing_active = True
                        self.timing_text = "Timing started"
                        print(f"Timing started (mode: {'real-time' if self.use_real_time_timing else 'frame-based'})")

                        if self.message_transmitter:
                            rel_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            self.message_transmitter.send_timing_start(self.video_name, rel_timestamp, self.frame_count)
                            self.state_machine.transition(Event.START, rel_timestamp)

                    elif self.timing_active:
                        if self.use_real_time_timing:
                            self.elapsed_time = current_time - self.start_time
                        else:
                            elapsed_frames = self.frame_count - self.start_frame
                            self.elapsed_time = elapsed_frames / self.video_fps

                        self.timing_active = False
                        self.timing_finished = True
                        self.timing_text = f"Time: {self.elapsed_time:.2f}s"
                        print(f"Timing ended: {self.elapsed_time:.2f} seconds (mode: {'real-time' if self.use_real_time_timing else 'frame-based'})")

                        self.exam_score = self.calculate_score(self.elapsed_time)

                        if self.message_transmitter:
                            rel_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            self.message_transmitter.send_timing_end(self.video_name, rel_timestamp, self.frame_count)
                            self.state_machine.transition(Event.END, rel_timestamp)

                    print(f"Line crossing detected: {self.line_crossing_text}")

                    break

            if self.line_crossing_cooldown > 0:
                break

        self.last_foot_positions = person_foot_points

        if self.timing_active:
            if self.use_real_time_timing:
                current_time = time.time()
                current_elapsed = current_time - self.start_time
            else:
                elapsed_frames = self.frame_count - self.start_frame
                current_elapsed = elapsed_frames / self.video_fps

            self.timing_text = f"Timing: {current_elapsed:.2f}s"

    def is_line_intersection(self, p1, p2, p3, p4):
        """Return True if segments p1-p2 and p3-p4 intersect."""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        
        return False
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame."""
        try:
            if frame is None:
                print("Warning: Received None frame")
                return self.display_frame if self.display_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            start_time = time.time()
            self.frame = frame.copy()
            processed_frame = frame.copy()

            try:
                self.last_process_time = time.time()

                max_process_time = 1.0

                if self.hoop_calibration_mode:
                    calibration_complete = self.shot_detector.calibrate_hoop_position(processed_frame)
                    if calibration_complete:
                        self.hoop_calibration_mode = False
                        print("Hoop calibration completed, switching to normal detection mode")
                    else:
                        self.hoop_calibration_frames_processed += 1
                        cv2.putText(processed_frame, f"Hoop Calibration: {self.hoop_calibration_frames_processed}/2",
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        return processed_frame

                pose_start = time.time()
                processed_frame = self.pose_detector.process_frame(processed_frame, self.frame_count)
                if time.time() - pose_start > max_process_time:
                    print("Warning: Pose detection timeout")
                    return self.display_frame if self.display_frame is not None else processed_frame

                hoop_start = time.time()
                processed_frame = self.shot_detector.process_frame(processed_frame, self.frame_count, self.pose_detector)
                if time.time() - hoop_start > max_process_time:
                    print("Warning: Hoop detection timeout")
                    return self.display_frame if self.display_frame is not None else processed_frame

                if not self.warmup_mode and not self.hoop_calibration_mode:
                    self.check_three_point_line_position()

                    self.check_line_crossing()

                    processed_frame = self.start_end_line_marker.process_frame(processed_frame)

                    shot_result = self.shot_detector.check_shot()
                    if shot_result:
                        self.handle_shot_event(shot_result)

                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)

                if (self.is_rtsp or self.is_camera) and self.frame_count % 30 == 0 and self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    print(f"Average processing time: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")

                self.frame_count += 1
                return processed_frame

            except Exception as e:
                print(f"Error processing frame: {e}")
                return processed_frame

        except Exception as e:
            print(f"Critical error in process_frame: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def check_three_point_line_position(self):
        """Update three-point in/out from foot points."""
        if len(self.shot_detector.three_point_marker.arc_points) < 3:
            return

        foot_points = self.pose_detector.get_foot_points()

        for person_idx, person_foot_points in enumerate(foot_points):
            prev_state = self.players_in_three_point_area.get(person_idx, True)
            is_out_of_bounds = self.shot_detector.three_point_marker.is_out_of_three_point_area(person_foot_points)

            current_state = not is_out_of_bounds

            if prev_state and not current_state:
                self.violation_text = "Out of Bounds!"
                self.violation_fade_counter = self.fade_frames

                if self.message_transmitter:
                    rel_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    self.message_transmitter.send_three_point_exit(self.video_name, rel_timestamp, self.frame_count)
                    self.state_machine.transition(Event.OUT_3PT, rel_timestamp)

            self.players_in_three_point_area[person_idx] = current_state

    def display_score(self, frame):
        """Score is composed in compose_display_frame."""
        pass

    def toggle_pose_detection(self):
        """Toggle pose pipeline flag."""
        self.enable_pose_detection = not self.enable_pose_detection
        return self.enable_pose_detection
    
    def start_three_point_marking(self):
        """Enter three-point marking."""
        if not self.marking_three_point:
            self.marking_three_point = True
            self.paused = True
            print("Entering three-point line marking mode")
            print("Left click: add point, Right click: finish, Middle click: cancel")
            self.display_frame = self.shot_detector.start_three_point_marking(self.frame)
            final_frame = self.compose_display_frame(
                self.display_frame,
                status_text="Three-Point Line Marking Mode",
                prompt_text=self.get_three_point_marking_prompt()
            )
            cv2.imshow('Frame', final_frame)
            cv2.waitKey(1)

    def start_start_end_lines_marking(self):
        """Enter start/end line marking."""
        if not self.marking_start_end_lines:
            self.marking_start_end_lines = True
            self.paused = True
            print("Entering start-end lines marking mode")
            print("Left click: add point, Right click: finish, Middle click: cancel")
            self.display_frame = self.start_end_line_marker.start_marking(self.frame)
            final_frame = self.compose_display_frame(
                self.display_frame,
                status_text="Start-End Lines Marking Mode",
                prompt_text=self.get_start_end_lines_marking_prompt()
            )
            cv2.imshow('Frame', final_frame)
            cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for marking modes."""
        if self.marking_three_point:
            if self.frame is not None:
                orig_x, orig_y = self.convert_display_to_original_coords(x, y, self.frame)
                if orig_x is not None and orig_y is not None:
                    self.shot_detector.three_point_mouse_callback(event, orig_x, orig_y, flags, param)

                    self.display_frame = self.shot_detector.process_frame(self.frame, self.frame_count, self.pose_detector)
                    final_frame = self.compose_display_frame(
                        self.display_frame,
                        status_text="Three-Point Line Marking Mode",
                        prompt_text=self.get_three_point_marking_prompt()
                    )
                    cv2.imshow('Frame', final_frame)
                    cv2.waitKey(1)

                    if not self.shot_detector.three_point_marker.is_marking:
                        self.marking_three_point = False
                        self.paused = False
                        print("Three-point line marking completed")

                        video_base_name = os.path.splitext(self.video_name)[0]
                        config_path = self._save_three_point_config_with_fallback(self.three_point_config_path, video_base_name)
                        if config_path:
                            print(f"Three-point line configuration saved to: {config_path}")
                        if getattr(self, 'use_parameter_mode', False):
                            if hasattr(self, 'need_three_point_marking'):
                                self.need_three_point_marking = False
                            self._rewind_video()
                            cv2.waitKey(1)

        elif self.marking_start_end_lines:
            if self.frame is not None:
                orig_x, orig_y = self.convert_display_to_original_coords(x, y, self.frame)
                if orig_x is not None and orig_y is not None:
                    self.start_end_line_marker.mouse_callback(event, orig_x, orig_y, flags, param)

                    self.display_frame = self.start_end_line_marker.draw_markers(self.frame)
                    final_frame = self.compose_display_frame(
                        self.display_frame,
                        status_text="Start-End Lines Marking Mode",
                        prompt_text=self.get_start_end_lines_marking_prompt()
                    )
                    cv2.imshow('Frame', final_frame)
                    cv2.waitKey(1)

                    if not self.start_end_line_marker.is_marking:
                        self.marking_start_end_lines = False
                        self.paused = False
                        print("Start-end lines marking completed")

                        video_base_name = os.path.splitext(self.video_name)[0]
                        config_path = self._save_start_end_config_with_fallback(self.start_end_config_path, video_base_name)
                        if config_path:
                            print(f"Start-end lines configuration saved to: {config_path}")
                        if getattr(self, 'use_parameter_mode', False):
                            if hasattr(self, 'need_start_end_marking'):
                                self.need_start_end_marking = False
                            self._rewind_video()
                            cv2.waitKey(1)

    def get_status_text(self):
        """Top bar status text."""
        if self.hoop_calibration_mode:
            return "Hoop Calibration"

        if not self.state_machine:
            return "No State Machine"

        progress = self.state_machine.get_progress()

        if progress["state"] == "Complete":
            return "[OK] Test Complete!"
        elif progress["state"] == "Initial":
            return "Ready to Start"
        elif progress["state"] == "First Shot":
            return "First Shot Phase"
        elif progress["state"] == "Retry Shot":
            return f"Retry Phase ({progress['retry_count']})"
        elif progress["state"] == "Out of Three-Point Line":
            return "Exit Three-Point Line"
        else:
            return progress["state"]

    def get_score_text(self):
        """Score line with state machine counts."""
        if not self.state_machine:
            return f"Shots: {self.shot_detector.makes}/{self.shot_detector.attempts}"
            
        progress = self.state_machine.get_progress()
        valid_shots = progress["valid_shots"].split("/")[0]
        
        return f"Valid Shots: {valid_shots}/4"

    def _load_score_config(self):
        """Load configs/score.json."""
        try:
            config_path = 'configs/score.json'
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.score_config = json.load(f)
                    print("Score config loaded")
            else:
                print(f"Warning: score config missing: {config_path}")
                self.score_config = None
        except Exception as e:
            print(f"Failed to load score config: {e}")
            self.score_config = None

    def calculate_score(self, time_seconds: float) -> Optional[int]:
        """Exam score from elapsed time and self.gender; 0 if too slow."""
        if not self.score_config or self.gender not in self.score_config:
            return None

        gender_config = self.score_config[self.gender]

        for score_str in sorted(gender_config.keys(), reverse=True):
            if time_seconds <= gender_config[score_str]:
                return int(float(score_str))

        return 0

    def get_timer_text(self):
        """Timer line for UI."""
        if not self.state_machine:
            return ""

        progress = self.state_machine.get_progress()
        if progress["state"] == "Complete":
            if self.timing_finished:
                return f"Final Time: {self.elapsed_time:.2f}s"
            return ""
        elif progress["state"] == "Initial":
            return ""
        else:
            return self.timing_text if (self.timing_active or self.timing_finished) else ""

    def get_prompt_text(self):
        """Bottom prompt for initial / complete / calibration."""
        if self.hoop_calibration_mode:
            return f"Calibrating hoop position...\nFrame {self.hoop_calibration_frames_processed}/2\nPlease ensure the basketball hoop is clearly visible"
            
        if not self.state_machine:
            return ""

        progress = self.state_machine.get_progress()

        if progress["state"] == "Complete":
            is_success, completion_msg = self.state_machine.get_completion_info()
            if is_success:
                if self.timing_finished:
                    prompt = f"Test Complete Successfully! Final Time: {self.elapsed_time:.2f}s"
                    if self.exam_score is not None:
                        prompt += f"\nScore: {self.exam_score}"
                    return prompt
                else:
                    return "Test Complete Successfully"
            else:
                return completion_msg

        elif progress["state"] == "Initial":
            return "Cross the start line to begin the test"
            
        else:
            return ""

    def compose_display_frame(self, raw_frame, status_text="", score_text="", timer_text="", prompt_text=""):
        """Layout 840x840 canvas: video center, top/bottom info bands."""
        canvas_w, canvas_h = 840, 840
        top_info_h = 100
        bottom_info_h = 150
        video_h = canvas_h - top_info_h - bottom_info_h
        video_w = canvas_w

        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        h, w = raw_frame.shape[:2]
        scale = min(video_w / w, video_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(raw_frame, (new_w, new_h))
        x0 = (video_w - new_w) // 2
        y0 = top_info_h + (video_h - new_h) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized

        cv2.rectangle(canvas, (0, 0), (canvas_w, top_info_h), (240,240,240), -1)
        cv2.rectangle(canvas, (0, canvas_h-bottom_info_h), (canvas_w, canvas_h), (240,240,240), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thick = 2

        if score_text:
            cv2.putText(canvas, score_text, (20, 40), font, font_scale, (0,0,0), font_thick, cv2.LINE_AA)

        if status_text:
            (status_w, _), _ = cv2.getTextSize(status_text, font, font_scale, font_thick)
            cv2.putText(canvas, status_text, ((canvas_w-status_w)//2, 40), font, font_scale, (0,0,0), font_thick, cv2.LINE_AA)

        if timer_text:
            (timer_w, _), _ = cv2.getTextSize(timer_text, font, font_scale, font_thick)
            cv2.putText(canvas, timer_text, (canvas_w-20-timer_w, 40), font, font_scale, (0,0,0), font_thick, cv2.LINE_AA)

        prompt_font_scale = 0.8
        prompt_thick = 2

        if prompt_text:
            lines = prompt_text.split('\n')
            y_offset = canvas_h - bottom_info_h + 40
            line_spacing = 40

            for line in lines:
                if line.strip():
                    (text_w, _), _ = cv2.getTextSize(line, font, prompt_font_scale, prompt_thick)
                    x_pos = (canvas_w - text_w) // 2
                    cv2.putText(canvas, line, (x_pos, y_offset), font, prompt_font_scale, (0,0,0), prompt_thick, cv2.LINE_AA)
                    y_offset += line_spacing

        return canvas

    def get_state_machine_text(self):
        """Next requirement / status string."""
        if not self.state_machine:
            return ""

        progress = self.state_machine.get_progress()
        requirement = self.state_machine.get_next_requirement()

        if progress["state"] == "Complete":
            if self.timing_finished:
                return f"Test Complete! Final Time: {self.elapsed_time:.2f}s"
            return "Test Complete! All requirements met."
        elif progress["state"] == "Initial":
            return "Ready to Start"
        elif progress["state"] == "Out of Three-Point Line":
            if progress["valid_shots"] == "4/4":
                return "Exit three-point line to complete test"
            elif not progress["can_start_new"]:
                return "Exit three-point line to continue"
            else:
                return requirement
        else:
            missing_requirements = []
            if progress["state"] == "Retry Shot":
                if int(progress["retry_count"].split("/")[0]) >= 2:
                    missing_requirements.append("Two retries failed - Exit three-point line")
            
            if missing_requirements:
                return " | ".join(missing_requirements)
            return requirement

    def handle_shot_event(self, shot_result: Dict) -> None:
        """Apply shot_result to state machine."""
        if not shot_result:
            return

        try:
            event_type = shot_result.get('event')
            if not event_type:
                print("Warning: Shot result missing event type")
                return

            current_time = time.time()

            try:
                event = Event[event_type.upper()]
            except KeyError:
                print(f"Warning: Invalid event type {event_type}")
                return

            try:
                success, error_msg = self.state_machine.transition(event, current_time)

                if success:
                    self.overlay_text = shot_result.get('text', 'Shot detected')
                    self.overlay_color = shot_result.get('color', (0, 255, 0))
                    self.fade_counter = self.fade_frames

                    if self.message_transmitter:
                        rel_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        self.message_transmitter.send_event(
                            self.video_name,
                            rel_timestamp,
                            event_type,
                            self.frame_count
                        )

                    if error_msg:
                        self.overlay_text = f"{self.overlay_text}\nWarning: {error_msg}"
                        self.overlay_color = (0, 165, 255)
                else:
                    print(f"Warning: State transition failed - {error_msg}")
                    self.overlay_text = f"Invalid Action: {error_msg}"
                    self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames * 2

            except Exception as transition_error:
                print(f"Error in state transition: {transition_error}")
                if self.message_transmitter:
                    self.message_transmitter.send_error(f"State transition error: {str(transition_error)}")

        except Exception as e:
            print(f"Error handling shot event: {e}")
            if self.message_transmitter:
                self.message_transmitter.send_error(f"Shot event handling error: {str(e)}")

    def handle_line_crossing_event(self, is_out_three_point: bool) -> None:
        """Handle OUT_3PT for state machine."""
        try:
            current_time = time.time()

            if is_out_three_point:
                success, error_msg = self.state_machine.transition(Event.OUT_3PT, current_time)

                if success:
                    next_requirement = self.state_machine.get_next_requirement()
                    self.overlay_text = f"Out of three-point line - {next_requirement}"

                    if error_msg:
                        self.overlay_text = f"{self.overlay_text}\nWarning: {error_msg}"
                        self.overlay_color = (0, 165, 255)
                    else:
                        self.overlay_color = (0, 255, 0)

                    self.fade_counter = self.fade_frames

                else:
                    self.overlay_text = f"Invalid Action: {error_msg}"
                    self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames * 2

        except Exception as e:
            print(f"Error handling line crossing event: {e}")
            if self.message_transmitter:
                self.message_transmitter.send_error(f"Line crossing event handling error: {str(e)}")

    def start_test(self) -> None:
        """Start a new test run."""
        try:
            current_time = time.time()

            self.state_machine = StateMachine()

            success = self.state_machine.transition(Event.START, current_time)

            if success:
                self.shot_detector.reset_shot_status()

                self.overlay_text = "Test Started - Attempt First Shot"
                self.overlay_color = (0, 255, 0)
                self.fade_counter = self.fade_frames

                self.timing_active = True
                if self.use_real_time_timing:
                    self.start_time = current_time
                else:
                    self.start_frame = self.frame_count
                    self.start_time = current_time
                self.elapsed_time = 0
                self.timing_finished = False

        except Exception as e:
            print(f"Error starting test: {e}")
            if self.message_transmitter:
                self.message_transmitter.send_error(f"Test start error: {str(e)}")

    def end_test(self) -> None:
        """End current test."""
        try:
            current_time = time.time()

            success = self.state_machine.transition(Event.END, current_time)

            if success:
                self.timing_active = False
                self.timing_finished = True

                is_complete = self.state_machine.is_complete()
                self.overlay_text = "Test Complete!" if is_complete else "Test Ended Prematurely"
                self.overlay_color = (0, 255, 0) if is_complete else (255, 165, 0)
                self.fade_counter = self.fade_frames
                
                
        except Exception as e:
            print(f"Error ending test: {e}")
            if self.message_transmitter:
                self.message_transmitter.send_error(f"Test end error: {str(e)}")

    def run(self) -> Optional[str]:
        """Main loop; returns output path if recording."""
        recording = False
        frame_skip_counter = 0
        max_skip_frames = 5

        try:
            source_info = "Camera" if self.is_camera else "RTSP stream" if self.is_rtsp else f"File: {self.video_name}"
            print(f"Source: {source_info}")

            use_parameter_mode = getattr(self, 'use_parameter_mode', False)

            if use_parameter_mode:
                recording = not self.is_camera and not self.is_rtsp
            else:
                try:
                    recording = self.is_rtsp or not self.is_camera or input("Record camera input? (y/n): ").lower() == 'y'
                except Exception as input_error:
                    print(f"Error getting recording preference: {input_error}")
                    recording = False

            need_rewind = False
            mark_three_point = False
            mark_start_end_lines = False

            if self.is_camera:
                print("Camera warming up...")
                warmup_frames = 10
                warmup_counter = 0

                self.warmup_mode = True

                while warmup_counter < warmup_frames:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Camera warmup failed: no frame")
                        break

                    self.frame = frame.copy()
                    self.display_frame = frame.copy()

                    progress_text = f"Warming up {warmup_counter + 1}/{warmup_frames}"
                    temp_frame = frame.copy()
                    cv2.putText(temp_frame, progress_text, (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Frame', temp_frame)
                    cv2.waitKey(1)

                    warmup_counter += 1
                    time.sleep(0.1)

                self.warmup_mode = False
                print("Camera warmup done.")

            print("Starting hoop calibration...")
            self.hoop_calibration_mode = True
            self.hoop_calibration_frames_processed = 0
            self.shot_detector.start_hoop_calibration()

            use_parameter_mode = getattr(self, 'use_parameter_mode', False)

            try:
                has_three_point_config = len(self.shot_detector.three_point_marker.arc_points) >= 3
                three_point_config_status = "exists" if has_three_point_config else "does not exist"
                print(f"Three-point line configuration {three_point_config_status}")

                if use_parameter_mode:
                    mark_three_point = getattr(self, 'need_three_point_marking', False)
                    if mark_three_point:
                        print("Auto three-point marking...")
                else:
                    try:
                        mark_three_point = input("Do you want to mark the three-point area? (y/n): ").lower() == 'y'
                    except Exception as input_error:
                        print(f"Error getting three-point marking preference: {input_error}")
                        mark_three_point = False

                if mark_three_point:
                    need_rewind = True
                    if self.frame is not None:
                        self.start_three_point_marking()
                    else:
                        ret, first_frame = self.cap.read()
                        if ret:
                            self.frame = first_frame.copy()
                            self.start_three_point_marking()
                        else:
                            print("Unable to read video frame for three-point marking")

                    while self.marking_three_point:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            return None

                    if use_parameter_mode and getattr(self, 'need_three_point_marking', False):
                        video_base_name = getattr(self, 'video_base_name', self.video_name)
                        config_path = self._save_three_point_config_with_fallback(self.three_point_config_path, video_base_name)
                        if config_path:
                            print(f"Three-point config saved: {config_path}")
                        self.need_three_point_marking = False
                        self._rewind_video()
                        cv2.waitKey(1)

                has_start_end_lines_config = len(self.start_end_line_marker.line_points) > 0
                start_end_lines_config_status = "exists" if has_start_end_lines_config else "does not exist"
                print(f"Start-end lines configuration {start_end_lines_config_status}")

                if use_parameter_mode:
                    mark_start_end_lines = getattr(self, 'need_start_end_marking', False)
                    if mark_start_end_lines:
                        print("Auto start-end line marking...")
                else:
                    try:
                        mark_start_end_lines = input("Do you want to mark the start-end lines? (y/n): ").lower() == 'y'
                    except Exception as input_error:
                        print(f"Error getting start-end lines marking preference: {input_error}")
                        mark_start_end_lines = False

                if mark_start_end_lines:
                    need_rewind = True
                    if self.frame is not None:
                        self.start_start_end_lines_marking()
                    else:
                        ret, first_frame = self.cap.read()
                        if ret:
                            self.frame = first_frame.copy()
                            self.start_start_end_lines_marking()
                        else:
                            print("Unable to read video frame for start-end lines marking")

                    while self.marking_start_end_lines:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            return None

                    if use_parameter_mode and getattr(self, 'need_start_end_marking', False):
                        video_base_name = getattr(self, 'video_base_name', self.video_name)
                        config_path = self._save_start_end_config_with_fallback(self.start_end_config_path, video_base_name)
                        if config_path:
                            print(f"Start-end config saved: {config_path}")
                        self.need_start_end_marking = False
                        self._rewind_video()
                        cv2.waitKey(1)

            except Exception as marking_error:
                print(f"Error during marking process: {marking_error}")
                if self.message_transmitter:
                    self.message_transmitter.send_error(f"Marking process error: {str(marking_error)}")

            if need_rewind and not self.is_camera and not self.is_rtsp:
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception as rewind_error:
                    print(f"Error rewinding video: {rewind_error}")
                    try:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.input_source)
                    except Exception as reopen_error:
                        print(f"Error reopening video: {reopen_error}")
                        return None

            self.fps_timer = time.time()
            self.fps_counter = 0
            self.fps = 0
            
            while True:
                if self.shutdown_requested:
                    print("Shutdown requested, exiting main loop...")
                    break

                try:
                    if not self.paused:
                        ret, frame = self.cap.read()
                        if not ret:
                            if self.is_camera:
                                print("Camera error or disconnected")
                                break
                            elif self.is_rtsp:
                                print("RTSP stream error or disconnected. Trying to reconnect...")
                                try:
                                    self.cap.release()
                                    time.sleep(2)
                                    self.cap = cv2.VideoCapture(self.input_source, cv2.CAP_FFMPEG)
                                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                                    ret, frame = self.cap.read()
                                    if not ret:
                                        print("Failed to reconnect to RTSP stream")
                                        break
                                    print("Successfully reconnected to RTSP stream")
                                except Exception as rtsp_error:
                                    print(f"Error reconnecting to RTSP stream: {rtsp_error}")
                                    break
                            else:
                                break

                        try:
                            frame_start_time = time.time()

                            if frame_skip_counter > 0 and frame_skip_counter < max_skip_frames:
                                frame_skip_counter += 1
                                continue

                            self.display_frame = self.process_frame(frame)

                            if self.hoop_calibration_mode:
                                cv2.putText(self.display_frame, f"Hoop Calibration: {self.hoop_calibration_frames_processed}/2",
                                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            process_time = time.time() - frame_start_time
                            if process_time > 0.5:
                                frame_skip_counter = 1
                                print(f"Warning: Long frame processing time ({process_time:.3f}s), skipping next frame")
                            else:
                                frame_skip_counter = 0

                        except Exception as process_error:
                            print(f"Error processing frame: {process_error}")
                            if self.frame is not None:
                                self.display_frame = self.frame.copy()
                            continue
                    else:
                        if self.frame is not None:
                            try:
                                self.display_frame = self.process_frame(self.frame)
                            except Exception as pause_process_error:
                                print(f"Error processing paused frame: {pause_process_error}")
                                self.display_frame = self.frame.copy()

                    try:
                        final_frame = self.compose_display_frame(
                            self.display_frame,
                            status_text=self.get_status_text(),
                            score_text=self.get_score_text(),
                            timer_text=self.get_timer_text(),
                            prompt_text=self.get_prompt_text()
                        )

                        if recording and final_frame is not None:
                            try:
                                if self.out is not None and self.out.isOpened():
                                    self.out.write(final_frame)
                                else:
                                    print("Warning: Video writer is not available")
                                    recording = False
                            except Exception as write_error:
                                print(f"Error writing frame to video: {write_error}")
                                recording = False
                            
                        cv2.imshow('Frame', final_frame)
                    except Exception as display_error:
                        print(f"Error displaying frame: {display_error}")
                        cv2.imshow('Frame', np.zeros((480, 640, 3), dtype=np.uint8))

                    try:
                        wait_time = 1 if not self.paused else 100
                        key = cv2.waitKey(wait_time) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('p'):
                            self.toggle_pose_detection()
                        elif key == ord(' '):
                            self.paused = not self.paused
                            frame_skip_counter = 0
                        elif key == ord('m'):
                            self.start_three_point_marking()
                        elif key == ord('l'):
                            self.start_start_end_lines_marking()
                        elif key == ord('r') and self.is_camera:
                            recording = not recording
                            print("Recording " + ("started" if recording else "stopped"))
                        elif key == ord('t'):
                            self.timing_active = False
                            self.timing_finished = False
                            self.timing_text = ""
                            self.start_time = 0
                            self.start_frame = 0
                            self.elapsed_time = 0
                            print("Timing reset")
                    except Exception as key_error:
                        print(f"Error processing keyboard input: {key_error}")
                        
                except Exception as loop_error:
                    print(f"Error in main processing loop: {loop_error}")
                    if self.message_transmitter:
                        self.message_transmitter.send_error(f"Main loop error: {str(loop_error)}")
                    continue

        except Exception as run_error:
            print(f"Critical error in run method: {run_error}")
            if self.message_transmitter:
                self.message_transmitter.send_error(f"Critical run error: {str(run_error)}")
                
        finally:
            try:
                if hasattr(self, 'state_machine') and self.state_machine:
                    is_success, completion_msg = self.state_machine.get_completion_info()
                    print(f"\n=== Test summary ===")
                    print(f"Status: {completion_msg}")
                    if self.timing_finished:
                        print(f"Final time: {self.elapsed_time:.2f}s")
                        if self.exam_score is not None:
                            print(f"Exam score: {self.exam_score}")
                    print(f"Valid makes: {self.state_machine.valid_shots}/4")
                    print("=" * 30)

                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()

                if recording and hasattr(self, 'out') and self.out is not None and self.out.isOpened():
                    if not self.shutdown_requested:
                        try:
                            print("Saving video in finally block...")
                            self.force_save_video()
                        except Exception as video_release_error:
                            print(f"Error releasing video writer in finally: {video_release_error}")
                            if hasattr(self, 'output_path'):
                                self._create_backup_info(f"Finally block error: {video_release_error}")

                cv2.destroyAllWindows()
                
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

        return self.output_path if recording else None

    def initialize_new_state(self) -> None:
        """Reset state machine and shot detector."""
        try:
            self.state_machine = StateMachine()
            self.shot_detector.reset_shot_status()
            self.timing_active = False
            self.timing_finished = False
            self.elapsed_time = 0
            self.start_time = 0
            self.start_frame = 0
            self.overlay_text = "Ready to start new test"
            self.overlay_color = (255, 255, 255)
            self.fade_counter = self.fade_frames
            
        except Exception as e:
            print(f"Error initializing new state: {e}")
            if self.message_transmitter:
                self.message_transmitter.send_error(f"State initialization error: {str(e)}")

    def signal_handler(self, signum, frame):
        """
        Handle shutdown signals (Ctrl+C, etc.)
        """
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        try:
            if hasattr(self, 'state_machine') and self.state_machine:
                is_success, completion_msg = self.state_machine.get_completion_info()
                print(f"\n=== Test summary ===")
                print(f"Status: {completion_msg}")
                if self.timing_finished:
                    print(f"Final time: {self.elapsed_time:.2f}s")
                    if self.exam_score is not None:
                        print(f"Exam score: {self.exam_score}")
                print(f"Valid makes: {self.state_machine.valid_shots}/4")
                print("=" * 30)

            self.force_save_video()
            time.sleep(0.1)
            cv2.destroyAllWindows()
            print("Graceful shutdown completed.")
            
        except Exception as e:
            print(f"Error during signal handling: {e}")
        sys.exit(0)

    def force_save_video(self):
        """Flush and release video writer."""
        try:
            if hasattr(self, 'out') and self.out is not None and self.out.isOpened():
                print("Finalizing video file...")

                try:
                    if hasattr(self, 'display_frame') and self.display_frame is not None:
                        for i in range(3):
                            final_frame = self.compose_display_frame(
                                self.display_frame,
                                status_text="Recording Stopped",
                                score_text="",
                                timer_text="",
                                prompt_text=f"Video saved successfully ({i+1}/3)"
                            )
                            self.out.write(final_frame)
                            time.sleep(0.01)
                except Exception as final_frame_error:
                    print(f"Warning: Could not write final frames: {final_frame_error}")

                try:
                    self.out.write(final_frame)
                    time.sleep(0.05)
                except Exception as flush_error:
                    print(f"Warning: Could not flush video buffer: {flush_error}")

                try:
                    self.out.release()
                    self.out = None
                    print(f"Video saved to {self.output_path}")
                except Exception as release_error:
                    print(f"Error releasing video writer: {release_error}")

                if os.path.exists(self.output_path):
                    file_size = os.path.getsize(self.output_path)
                    print(f"Video file size: {file_size} bytes")

                    if file_size < 1024:
                        print("Warning: Video file size is very small, may be corrupted")
                        self._create_backup_info("File size too small")
                        return

                    try:
                        test_cap = cv2.VideoCapture(self.output_path)
                        if test_cap.isOpened():
                            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = test_cap.get(cv2.CAP_PROP_FPS)
                            duration = frame_count / fps if fps > 0 else 0
                            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            print(f"Video verification successful:")
                            print(f"  - Frames: {frame_count}")
                            print(f"  - FPS: {fps:.2f}")
                            print(f"  - Duration: {duration:.2f}s")
                            print(f"  - Resolution: {width}x{height}")

                            ret, frame = test_cap.read()
                            if ret and frame is not None:
                                print("  - First frame readable: OK")
                            else:
                                print("  - Warning: Could not read first frame")
                                self._create_backup_info("First frame not readable")
                        else:
                            print("Warning: Could not open saved video file for verification")
                            self._create_backup_info("Could not open video file")
                        
                        test_cap.release()
                        
                    except Exception as verify_error:
                        print(f"Warning: Could not verify video file: {verify_error}")
                        self._create_backup_info(f"Verification error: {verify_error}")
                else:
                    print("Warning: Video file not found after saving")
                    self._create_backup_info("Video file not found")
            else:
                print("Warning: Video writer not available or not opened")
                self._create_backup_info("Video writer not available")
        except Exception as e:
            print(f"Error during force save: {e}")
            self._create_backup_info(f"Force save error: {str(e)}")

    def _create_backup_info(self, error_msg):
        """Create backup information file when video save fails"""
        try:
            if hasattr(self, 'output_path'):
                backup_path = self.output_path.replace('.mp4', '_backup_info.txt').replace('.avi', '_backup_info.txt')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(f"Video recording interrupted at: {datetime.now()}\n")
                    f.write(f"Original output path: {self.output_path}\n")
                    f.write(f"Frame count processed: {self.frame_count}\n")
                    f.write(f"Error during save: {error_msg}\n")
                    f.write(f"Processing time: {time.time() - getattr(self, 'last_process_time', time.time()):.2f}s\n")
                print(f"Backup info saved to: {backup_path}")
        except Exception as backup_error:
            print(f"Could not create backup info: {backup_error}")

    def convert_display_to_original_coords(self, display_x, display_y, original_frame):
        """Map canvas click to source frame coords."""
        canvas_w, canvas_h = 840, 840
        top_info_h = 100
        bottom_info_h = 150
        video_h = canvas_h - top_info_h - bottom_info_h
        video_w = canvas_w

        orig_h, orig_w = original_frame.shape[:2]

        scale = min(video_w / orig_w, video_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        x_offset = (video_w - new_w) // 2
        y_offset = top_info_h + (video_h - new_h) // 2

        video_x = display_x - x_offset
        video_y = display_y - y_offset

        if video_x < 0 or video_x >= new_w or video_y < 0 or video_y >= new_h:
            return None, None

        orig_x = int(video_x / scale)
        orig_y = int(video_y / scale)
        
        return orig_x, orig_y

    def get_three_point_marking_prompt(self):
        """Three-point marking help line."""
        if not self.marking_three_point:
            return ""
        
        marker = self.shot_detector.three_point_marker
        return f"Mark polygon points ({len(marker.arc_points)})\nLeft: add point | Right: finish | Middle: cancel"
    
    def get_start_end_lines_marking_prompt(self):
        """Start-end marking help line."""
        if not self.marking_start_end_lines:
            return ""
        
        marker = self.start_end_line_marker
        current_line_progress = f" - Points: {len(marker.current_line)}/2" if len(marker.current_line) > 0 else ""
        return f"Mark lines ({len(marker.line_points)}/3){current_line_progress}\nLeft: add point | Right: finish | Middle: cancel"