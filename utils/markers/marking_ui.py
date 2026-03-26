import cv2
import numpy as np

class MarkingUI:
    """Overlay text for polygon marking (instructions + transient errors)."""
    
    def __init__(self):
        self.error_message = ""
        self.error_frames = 0
        
    def reset(self):
        self.error_message = ""
        self.error_frames = 0
    
    def set_error(self, message, frames=60):
        self.error_message = message
        self.error_frames = frames
        print(self.error_message)
    
    def draw_simple_marking_ui(self, frame, arc_points):
        result = frame.copy()
        
        instructions = [
            "Click to add polygon points",
            f"Points marked: {len(arc_points)}",
            "Press ESC when done"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 30
            cv2.putText(result, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.error_message and self.error_frames > 0:
            text_bg = np.zeros((50, result.shape[1], 3), dtype=np.uint8)
            text_bg[:] = (0, 0, 255)
            result[150:200, 0:result.shape[1]] = cv2.addWeighted(
                result[150:200, 0:result.shape[1], :], 0.3, text_bg, 0.7, 0)
            
            cv2.putText(result, self.error_message, (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self.error_frames -= 1
        
        return result
