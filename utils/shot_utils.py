import math
import numpy as np
import cv2
import os
from datetime import datetime

DEBUG = False

def save_debug_rim_images(rim_region_base, occluded_frames_list, debug_dir, shot_frames=None):
    """Save 10x-upscaled rim crops and optional full frames for occlusion debugging."""
    try:
        if rim_region_base is not None and rim_region_base.size > 0:
            base_region_resized = cv2.resize(rim_region_base, 
                                            (rim_region_base.shape[1] * 10, rim_region_base.shape[0] * 10),
                                            interpolation=cv2.INTER_NEAREST)
            base_path = os.path.join(debug_dir, "00_base_rim_region.png")
            cv2.imwrite(base_path, base_region_resized)
        
        for frame_index, rim_region_occluded in occluded_frames_list:
            if rim_region_occluded is not None and rim_region_occluded.size > 0:
                occluded_region_resized = cv2.resize(rim_region_occluded,
                                                    (rim_region_occluded.shape[1] * 10, rim_region_occluded.shape[0] * 10),
                                                    interpolation=cv2.INTER_NEAREST)
                occluded_path = os.path.join(debug_dir, f"{frame_index:03d}_occluded_rim_region.png")
                cv2.imwrite(occluded_path, occluded_region_resized)
                
                if shot_frames is not None and frame_index < len(shot_frames):
                    original_frame = shot_frames[frame_index]
                    original_frame_path = os.path.join(debug_dir, f"{frame_index:03d}_original_frame.png")
                    cv2.imwrite(original_frame_path, original_frame)
        
        return debug_dir
        
    except Exception as e:
        print(f"Debug image save error: {e}")
        return None


def check_hoop_rim_occlusion(shot_frames, hoop_pos, ball_color, ball_pos=None):
    """
    Heuristic: ball-colored pixels on the hoop rim strip imply rim occlusion.
    hoop_pos/ball_pos entries: ((x,y), frame_count, w, h, confidence).
    """
    
    if len(shot_frames) < 2 or len(hoop_pos) < 1:
        return False
    
    hoop_center = hoop_pos[-1][0]
    hoop_center_x, hoop_center_y = int(hoop_center[0]), int(hoop_center[1])
    hoop_width = int(hoop_pos[-1][2])
    hoop_height = int(hoop_pos[-1][3])
    
    hoop_x1_full = hoop_center_x - hoop_width // 2
    hoop_y1 = hoop_center_y - hoop_height // 2
    rim_line_height = 5
    
    trim_ratio = 0.15
    rim_x1 = hoop_x1_full + int(hoop_width * trim_ratio)
    rim_width = int(hoop_width * (1 - 2 * trim_ratio))
    
    rim_y1_trimmed = hoop_y1 + 1
    rim_height_trimmed = rim_line_height - 2
    
    if rim_x1 < 0 or rim_y1_trimmed < 0:
        return False
    
    if len(shot_frames) == 0:
        return False
    
    first_frame = shot_frames[0]
    frame_h, frame_w = first_frame.shape[:2]
    
    if rim_x1 + rim_width > frame_w or rim_y1_trimmed + rim_height_trimmed > frame_h:
        return False
    
    rim_region_base = first_frame[rim_y1_trimmed:rim_y1_trimmed + rim_height_trimmed, 
                                   rim_x1:rim_x1 + rim_width]
    if rim_region_base.size == 0:
        return False
    
    debug_dir = None
    if DEBUG:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        debug_dir = f"debug/shot/{timestamp}"
        os.makedirs(debug_dir, exist_ok=True)
    
    occluded_frames_list = []
    
    ball_color_similarity_threshold = 40
    occlusion_pixel_ratio_threshold = 0.1
    
    if isinstance(ball_color, (list, tuple)):
        ball_color_array = np.array(ball_color, dtype=np.float32).reshape(1, 1, 3)
    else:
        ball_color_array = np.array(ball_color, dtype=np.float32).reshape(1, 1, 3)
    
    for i in range(1, len(shot_frames)):
        frame = shot_frames[i]
        frame_h, frame_w = frame.shape[:2]
        
        if rim_x1 + rim_width > frame_w or rim_y1_trimmed + rim_height_trimmed > frame_h:
            continue
        
        rim_region_current = frame[rim_y1_trimmed:rim_y1_trimmed + rim_height_trimmed,
                                    rim_x1:rim_x1 + rim_width]
        
        if rim_region_current.size == 0:
            continue
        
        color_diff = np.sqrt(np.sum((rim_region_current.astype(np.float32) - ball_color_array) ** 2, axis=2))
        
        pixels_similar_to_ball = np.sum(color_diff < ball_color_similarity_threshold)
        total_pixels = color_diff.size
        similarity_ratio = pixels_similar_to_ball / total_pixels if total_pixels > 0 else 0
        
        if similarity_ratio > occlusion_pixel_ratio_threshold:
            should_ignore_all = False
            if ball_pos is not None and i < len(ball_pos):
                ball_data = ball_pos[i]
                if ball_data is not None:
                    ball_center = ball_data[0]
                    ball_height = ball_data[3]
                    
                    hoop_rim_top = rim_y1_trimmed
                    
                    if (ball_center[1] + ball_height < hoop_rim_top):
                        should_ignore_all = True
                        occluded_frames_list.clear()
            
            if not should_ignore_all:
                occluded_frames_list.append((i, rim_region_current.copy()))
    
    is_occluded = len(occluded_frames_list) > 0
    
    if DEBUG:
        save_debug_rim_images(rim_region_base, occluded_frames_list, debug_dir, shot_frames)
    
    return is_occluded


def score(ball_pos, hoop_pos, shot_frames=None):
    """
    True if the ball path passes close to the hoop center (segment distance test).
    Optional rim occlusion check can veto a make when the ball moves down.
    """
    
    if len(ball_pos) < 2 or len(hoop_pos) < 1:
        return False
    
    if shot_frames is not None and len(shot_frames) > 0:
        if len(ball_pos) >= 2:
            ball_moving_down = False
            for i in range(max(0, len(ball_pos) - 3), len(ball_pos)):
                if i > 0:
                    prev_y = ball_pos[i-1][0][1]
                    curr_y = ball_pos[i][0][1]
                    if curr_y > prev_y:
                        ball_moving_down = True
                        break
            
            ball_color = None
            if len(ball_pos) > 0 and len(shot_frames) > 0:
                ball_colors = []
                for i in range(min(3, len(ball_pos), len(shot_frames))):
                    ball_data = ball_pos[i]
                    frame = shot_frames[i]
                    if ball_data is not None and frame is not None:
                        ball_center = ball_data[0]
                        ball_width = ball_data[2]
                        ball_height = ball_data[3]
                        
                        ball_x1 = int(ball_center[0] - ball_width // 2)
                        ball_y1 = int(ball_center[1] - ball_height // 2)
                        ball_x2 = int(ball_center[0] + ball_width // 2)
                        ball_y2 = int(ball_center[1] + ball_height // 2)
                        
                        frame_h, frame_w = frame.shape[:2]
                        if ball_x1 >= 0 and ball_y1 >= 0 and ball_x2 < frame_w and ball_y2 < frame_h:
                            ball_region = frame[ball_y1:ball_y2, ball_x1:ball_x2]
                            if ball_region.size > 0:
                                h, w = ball_region.shape[:2]
                                center_h, center_w = h // 2, w // 2
                                quarter_h, quarter_w = h // 4, w // 4
                                
                                ball_region_center = ball_region[quarter_h:center_h + quarter_h, 
                                                                 quarter_w:center_w + quarter_w]
                                
                                if ball_region_center.size > 0:
                                    avg_color = np.mean(ball_region_center.reshape(-1, 3), axis=0)
                                    ball_colors.append(avg_color)
                
                if len(ball_colors) > 0:
                    ball_color = np.mean(ball_colors, axis=0)
            
            if ball_moving_down and ball_color is not None:
                if check_hoop_rim_occlusion(shot_frames, hoop_pos, ball_color, ball_pos):
                    return False
    
    hoop_center = hoop_pos[-1][0]
    hoop_center_y = hoop_center[1]
    
    point_a = None
    point_a_index = -1
    
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < hoop_center_y:
            point_a = ball_pos[i][0]
            point_a_index = i
            break
    
    if point_a is None:
        return False
    
    point_b = None
    if point_a_index + 1 < len(ball_pos):
        point_b = ball_pos[point_a_index + 1][0]
    
    if point_b is None:
        return False
    
    distance = point_to_line_distance(point_a, point_b, hoop_center)
    
    return distance < 20


def point_to_line_distance(point_a, point_b, target_point):
    """Shortest distance from ``target_point`` to segment ab."""
    line_vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    target_vector = (target_point[0] - point_a[0], target_point[1] - point_a[1])
    line_length_sq = line_vector[0]**2 + line_vector[1]**2
    if line_length_sq < 1e-6:
        return math.sqrt(target_vector[0]**2 + target_vector[1]**2)
    
    t = max(0, min(1, (target_vector[0] * line_vector[0] + target_vector[1] * line_vector[1]) / line_length_sq))
    closest_point = (
        point_a[0] + t * line_vector[0],
        point_a[1] + t * line_vector[1]
    )
    distance = math.sqrt((target_point[0] - closest_point[0])**2 + (target_point[1] - closest_point[1])**2)
    
    return distance


def detect_down(ball_pos, hoop_pos):
    """True if ball center is below the hoop lower boundary (shot attempt heuristic)."""
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return False
    
    hoop_lower_boundary = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    
    ball_center_y = ball_pos[-1][0][1]
    return ball_center_y > hoop_lower_boundary


def detect_up(ball_pos, hoop_pos):
    """True if ball center is above the hoop lower boundary."""
    if len(ball_pos) < 1 or len(hoop_pos) < 1:
        return True
    
    hoop_lower_boundary = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    
    ball_center_y = ball_pos[-1][0][1]
    return ball_center_y < hoop_lower_boundary


def in_hoop_region(center, hoop_pos):
    """Loose axis-aligned region around the last hoop box."""
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def clean_ball_pos(ball_pos, frame_count):
    """Drop outliers: huge jump in few frames, or very elongated box; trim old tail."""
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """Drop impossible hoop jumps or bad aspect ratio; cap history length."""
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos
