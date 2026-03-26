import cv2
import numpy as np

class AreaBuilder:
    """Build and render a filled polygon overlay (e.g. three-point zone)."""
    
    @staticmethod
    def draw_simple_polygon_area(frame, polygon_points, display_points_and_lines=True):
        """Draw a semi-transparent red fill and optional blue outline."""
        result = frame.copy()
        
        if len(polygon_points) < 3:
            return result
            
        points_array = np.array(polygon_points, dtype=np.int32)
        mask = np.zeros(result.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points_array], (0, 0, 255))
        result = cv2.addWeighted(result, 1, mask, 0.3, 0)
        
        if display_points_and_lines:
            cv2.polylines(result, [points_array], True, (255, 0, 0), 2)
                    
        return result
