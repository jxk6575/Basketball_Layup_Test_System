import cv2
import numpy as np
from typing import List, Tuple
import json
import os
from .marking_ui import MarkingUI
from .area_builder import AreaBuilder

class ThreePointMarker:
    def __init__(self):
        self.arc_points = []
        self.is_marking = False
        self.current_frame = None
        self.config_dir = "configs/three_point_areas"
        self.need_redraw = True
        
        self.debug_mode = False
        
        self.ui = MarkingUI()
        
        os.makedirs(self.config_dir, exist_ok=True)

    def reset(self):
        self.arc_points = []
        self.is_marking = False
        self.need_redraw = True
        self.ui.reset()
    
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
            self.arc_points.append((x, y))
            print(f"Added polygon point {len(self.arc_points)} at ({x}, {y})")
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            action_taken = True
            if self.arc_points:
                removed_point = self.arc_points.pop()
                print(f"Removed last point at {removed_point}")
            else:
                self.ui.set_error("No points to remove", 30)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            action_taken = True
            if len(self.arc_points) >= 3:
                print(f"Polygon marking completed with {len(self.arc_points)} points")
                self.is_marking = False
            else:
                self.ui.set_error("Need at least 3 points to complete polygon", 60)
        
        elif flags & cv2.EVENT_FLAG_CTRLKEY and event == cv2.EVENT_LBUTTONDOWN:
            action_taken = True
            print("Polygon marking canceled")
            self.reset()
        
        if action_taken:
            self.need_redraw = True
    
    def draw_markers(self, frame: np.ndarray, show_points_and_lines: bool = True) -> np.ndarray:
        result = frame.copy()
        
        display_points_and_lines = self.is_marking or show_points_and_lines
        
        if display_points_and_lines:
            for i, point in enumerate(self.arc_points):
                cv2.circle(result, point, 5, (255, 0, 0), -1)
                cv2.putText(result, f"P{i+1}", (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            if len(self.arc_points) >= 3:
                for i in range(len(self.arc_points)):
                    start_point = self.arc_points[i]
                    end_point = self.arc_points[(i + 1) % len(self.arc_points)]
                    cv2.line(result, start_point, end_point, (255, 0, 0), 2)
        
        if len(self.arc_points) >= 3:
            try:
                result = AreaBuilder.draw_simple_polygon_area(result, self.arc_points, display_points_and_lines)
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"Exception in draw_markers: {e}")
                self.ui.set_error(f"Drawing error: {str(e)}", 60)
        
        if self.is_marking:
            result = self.ui.draw_simple_marking_ui(result, self.arc_points)
        
        self.need_redraw = False
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.is_marking:
            self.current_frame = frame.copy()
            if self.need_redraw:
                return self.draw_markers(frame)
            else:
                return self.draw_markers(self.current_frame)
        elif len(self.arc_points) >= 3:
            return self.draw_markers(frame, show_points_and_lines=False)
        else:
            return frame
    
    def save_config(self, video_name: str) -> str:
        if len(self.arc_points) < 3:
            print("No three-point area defined to save (need at least 3 points)")
            return ""
        
        config_name = f"{self.config_dir}/{video_name}_three_point.json"
            
        config = {
            "arc_points": self.arc_points,
            "version": "simplified"
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
        config_name = f"{self.config_dir}/{video_name}_three_point.json"
        
        if not os.path.exists(config_name):
            print(f"Configuration file not found: {config_name}")
            return False
            
        try:
            with open(config_name, 'r') as f:
                config = json.load(f)
            
            if "version" in config and config["version"] == "simplified":
                self.arc_points = config["arc_points"]
                print(f"Simplified configuration loaded from {config_name}")
                return True
            else:
                self.arc_points = config.get("arc_points", [])
                print(f"Legacy configuration loaded from {config_name} (arc_points only)")
                return True
                
        except Exception as e:
            print(f"Error loading config: {e}")
            return False

    def is_point_in_three_point_area(self, point: Tuple[int, int]) -> bool:
        if len(self.arc_points) < 3:
            return False

        try:
            polygon_points = np.array(self.arc_points, dtype=np.float32)
            result = cv2.pointPolygonTest(polygon_points, point, False)
            return result >= 0
            
        except Exception as e:
            if self.debug_mode:
                print(f"Exception in is_point_in_three_point_area: {e}")
            return False
    
    def is_point_above_front_line(self, point: Tuple[int, int]) -> bool:
        """True if point is in the x-span of the first two arc points and above that segment (smaller y)."""
        if len(self.arc_points) < 2:
            return False
            
        try:
            p1 = self.arc_points[0]
            p2 = self.arc_points[1]
            
            x, y = point
            x1, y1 = p1
            x2, y2 = p2
            
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            x_in_range = x_min <= x <= x_max
            
            y_max = max(y1, y2)
            y_above = y < y_max
            
            return x_in_range and y_above
                
        except Exception as e:
            if self.debug_mode:
                print(f"Exception in is_point_above_front_line: {e}")
            return False

    def is_out_of_three_point_area(self, foot_points: List[Tuple[int, int]]) -> bool:
        """True if any foot is outside the polygon and not counted as above the front line."""
        if len(self.arc_points) < 3 or not foot_points:
            return False
            
        try:
            for foot_point in foot_points:
                if not self.is_point_in_three_point_area(foot_point):
                    if not self.is_point_above_front_line(foot_point):
                        return True
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Exception in is_out_of_three_point_area: {e}")
            return False
