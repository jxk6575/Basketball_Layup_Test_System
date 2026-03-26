import cv2
import numpy as np
import json
import os

class StartEndLineMarker:
    """Mark and draw up to three start/end lines on video frames."""
    
    def __init__(self):
        self.line_points = []
        self.is_marking = False
        self.current_line = []
        self.current_frame = None
        self.config_dir = "configs/start_end_lines"
        self.need_redraw = True
        self.error_message = ""
        self.error_frames = 0
        
        self.line_colors = [
            (0, 255, 0),
            (0, 255, 0),
            (0, 255, 0)
        ]
        
        os.makedirs(self.config_dir, exist_ok=True)
    
    def reset(self):
        self.line_points = []
        self.is_marking = False
        self.current_line = []
        self.need_redraw = True
        self.error_message = ""
        self.error_frames = 0
    
    def start_marking(self, frame: np.ndarray):
        self.reset()
        self.current_frame = frame.copy()
        self.is_marking = True
        self.need_redraw = True
        return self.draw_markers(self.current_frame)
    
    def mouse_callback(self, event, x, y, flags, param):
        if not self.is_marking:
            return
            
        action_taken = False
        
        if event == cv2.EVENT_LBUTTONDOWN:
            action_taken = True
            if len(self.current_line) < 2:
                self.current_line.append((x, y))
                print(f"Added point {len(self.current_line)} at ({x}, {y}) to line {len(self.line_points) + 1}")
                
                if len(self.current_line) == 2:
                    self.line_points.append(self.current_line)
                    self.current_line = []
                    print(f"Line {len(self.line_points)} completed")
                    
                    if len(self.line_points) == 3:
                        print("All three lines completed")
                        self.is_marking = False
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            action_taken = True
            if len(self.current_line) == 0 and len(self.line_points) > 0:
                print(f"Finished marking with {len(self.line_points)} lines")
                self.is_marking = False
            elif len(self.current_line) > 0:
                self.set_error("Please complete the current line or clear it with middle click", 60)
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            action_taken = True
            if len(self.current_line) > 0:
                self.current_line = []
                print("Current line canceled")
            elif len(self.line_points) > 0:
                self.line_points = []
                print("All lines canceled")
            else:
                self.is_marking = False
                print("Marking canceled")
        
        if action_taken:
            self.need_redraw = True
    
    def set_error(self, message, frames=60):
        self.error_message = message
        self.error_frames = frames
        print(self.error_message)
    
    def draw_markers(self, frame: np.ndarray, show_points: bool = True) -> np.ndarray:
        result = frame.copy()
        
        for i, line in enumerate(self.line_points):
            if len(line) == 2:
                color = self.line_colors[i % len(self.line_colors)]
                cv2.line(result, line[0], line[1], color, 2)
                
                if show_points:
                    for j, point in enumerate(line):
                        cv2.circle(result, point, 5, color, -1)
                        cv2.putText(result, f"L{i+1}P{j+1}", (point[0]+10, point[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if len(self.current_line) > 0:
            color = self.line_colors[len(self.line_points) % len(self.line_colors)]
            
            if show_points:
                cv2.circle(result, self.current_line[0], 5, color, -1)
                cv2.putText(result, f"L{len(self.line_points)+1}P1", (self.current_line[0][0]+10, self.current_line[0][1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if self.error_message and self.error_frames > 0:
            text_bg = np.zeros((50, result.shape[1], 3), dtype=np.uint8)
            text_bg[:] = (0, 0, 255)
            result[100:150, 0:result.shape[1]] = cv2.addWeighted(
                result[100:150, 0:result.shape[1], :], 0.3, text_bg, 0.7, 0)
            
            cv2.putText(result, self.error_message, (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self.error_frames -= 1
        
        self.need_redraw = False
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.is_marking:
            self.current_frame = frame.copy()
            if self.need_redraw:
                return self.draw_markers(frame)
            else:
                return self.draw_markers(self.current_frame)
        elif len(self.line_points) > 0:
            return self.draw_markers(frame, show_points=False)
        else:
            return frame
    
    def save_config(self, video_name: str) -> str:
        if len(self.line_points) == 0:
            print("No lines defined to save")
            return ""
        
        config_name = f"{self.config_dir}/{video_name}_start_end_lines.json"
            
        config = {
            "line_points": self.line_points
        }
        
        try:
            with open(config_name, 'w') as f:
                json.dump(config, f)
            print(f"Configuration saved to {config_name}")
            return config_name
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return ""
    
    def load_config(self, video_name: str) -> bool:
        config_name = f"{self.config_dir}/{video_name}_start_end_lines.json"
        
        if not os.path.exists(config_name):
            print(f"Configuration file not found: {config_name}")
            return False
            
        try:
            with open(config_name, 'r') as f:
                config = json.load(f)
                
            self.line_points = config["line_points"]
            
            print(f"Configuration loaded from {config_name}")
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
