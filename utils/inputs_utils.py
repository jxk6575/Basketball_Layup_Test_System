import os
import glob
import cv2
import json
from enum import Enum
from typing import List, Dict, Optional

class InputType(Enum):
    FILE = "file"
    CAMERA = "camera"
    URL = "url"
    RTSP = "rtsp"
    
class InputSource:
    def __init__(self, source_path: str, source_type: InputType):
        self.source_path = source_path
        self.source_type = source_type
        self.name = os.path.basename(source_path) if source_type == InputType.FILE else source_path
        
    def get_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.source_path)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.source_type.value})"

def list_input_videos(input_dir: str = "inputs") -> List[InputSource]:
    """List video files under ``input_dir`` as FILE sources."""
    os.makedirs(input_dir, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(f'{input_dir}/*{ext}'))
    
    return [InputSource(file, InputType.FILE) for file in video_files]

def list_available_cameras() -> List[InputSource]:
    """Probe indices 0..19 and return cameras that open and read one frame."""
    available_cameras = []
    max_cameras_to_check = 20
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                camera_info = f"Camera {i} ({width}x{height}@{fps}fps)"
                available_cameras.append(InputSource(str(i), InputType.CAMERA))
                print(f"Found: {camera_info}")
            
            cap.release()
    
    print(f"Found {len(available_cameras)} camera(s)")
    return available_cameras

def list_rtsp_streams() -> List[InputSource]:
    """Load RTSP URLs from ``configs/rtsp_streams.json``."""
    rtsp_streams = []
    config_file = 'configs/rtsp_streams.json'
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                streams = json.load(f)
                for name, url in streams.items():
                    rtsp_streams.append(InputSource(url, InputType.RTSP))
                    print(f"Found RTSP stream: {name} - {url}")
        except json.JSONDecodeError:
            print("Error reading RTSP configuration file")
    
    print(f"Found {len(rtsp_streams)} RTSP stream(s)")
    return rtsp_streams

def add_rtsp_stream() -> Optional[InputSource]:
    """Prompt for name/URL, optional probe, save to config. Returns source or None."""
    os.makedirs('configs', exist_ok=True)
    
    config_file = 'configs/rtsp_streams.json'
    streams = {}
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                streams = json.load(f)
        except json.JSONDecodeError:
            print("Error reading RTSP config file")
    
    print("\nAdd RTSP stream (empty name to cancel):")
    stream_name = input("Stream name: ").strip()
    if not stream_name:
        return None
    
    stream_url = input("RTSP URL: ").strip()
    if not stream_url:
        return None
    
    if not stream_url.startswith("rtsp://"):
        print("Warning: URL does not start with 'rtsp://'")
        confirm = input("Continue anyway? (y/n): ").lower()
        if confirm != 'y':
            return None
    
    print("Testing RTSP connection...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: could not open RTSP stream")
        cap.release()
        confirm = input("Save anyway? (y/n): ").lower()
        if confirm != 'y':
            return None
    else:
        ret, _ = cap.read()
        if not ret:
            print("Warning: opened but failed to read a frame")
        else:
            print("RTSP stream OK")
        cap.release()
    
    streams[stream_name] = stream_url
    with open(config_file, 'w') as f:
        json.dump(streams, f, indent=4)
    
    print(f"Saved RTSP stream '{stream_name}'")
    return InputSource(stream_url, InputType.RTSP)

def get_input_sources() -> Dict[InputType, List[InputSource]]:
    """Group FILE / CAMERA / RTSP sources."""
    sources = {
        InputType.FILE: list_input_videos(),
        InputType.CAMERA: list_available_cameras(),
        InputType.RTSP: list_rtsp_streams(),
    }
    return sources

def select_input_source() -> Optional[InputSource]:
    """Interactive menu to pick an input source."""
    all_sources = get_input_sources()
    
    sources_list = []
    for source_type, sources in all_sources.items():
        if sources or source_type == InputType.RTSP:
            print(f"\n{source_type.value.upper()} sources:")
            for i, source in enumerate(sources, start=len(sources_list) + 1):
                print(f"{i}. {source}")
                sources_list.append(source)
            
            if source_type == InputType.RTSP:
                print(f"{len(sources_list) + 1}. [+] Add RTSP stream")
                sources_list.append("add_rtsp")
    
    if not sources_list:
        print("Error: no input sources found")
        return None
    
    while True:
        try:
            choice = input("\nSelect source number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return None
                
            choice_index = int(choice)
            if 1 <= choice_index <= len(sources_list):
                selected = sources_list[choice_index - 1]
                
                if selected == "add_rtsp":
                    return add_rtsp_stream()
                
                selected_source = selected
                
                if selected_source.source_type == InputType.CAMERA:
                    cap = cv2.VideoCapture(int(selected_source.source_path))
                    if not cap.isOpened():
                        print("Camera no longer available")
                        cap.release()
                        continue
                    cap.release()
                
                if selected_source.source_type == InputType.RTSP:
                    print("Testing RTSP connection...")
                    cap = cv2.VideoCapture(selected_source.source_path)
                    if not cap.isOpened():
                        print("Error: could not open RTSP stream")
                        cap.release()
                        confirm = input("Retry? (y/n): ").lower()
                        if confirm == 'y':
                            continue
                    else:
                        ret, _ = cap.read()
                        if not ret:
                            print("Warning: opened but failed to read a frame")
                        cap.release()
                
                return selected_source
            else:
                print("Invalid choice, try again")
        except ValueError:
            print("Enter a valid number")

def get_video_source(args=None) -> Optional[InputSource]:
    """CLI ``args.rtsp_url`` or interactive ``select_input_source``."""
    if args is not None:
        if hasattr(args, 'rtsp_url') and args.rtsp_url:
            print(f"Using RTSP: {args.rtsp_url}")
            return InputSource(args.rtsp_url, InputType.RTSP)
    return select_input_source()
