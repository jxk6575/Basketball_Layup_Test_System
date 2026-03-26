import os
import argparse
from detectors.detector_manager import DetectorManager
from utils.inputs_utils import get_video_source, InputType, InputSource
from utils.transmit_utils import MessageTransmitter
from utils.device_utils import get_device
import traceback
import signal
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

current_detector_manager = None
current_message_transmitter = None

def signal_handler(sig, frame):
    """Ctrl+C: flush video writer and disconnect."""
    print('\nInterrupt received, shutting down...')

    if current_detector_manager:
        print('Saving video and releasing capture...')
        try:
            current_detector_manager.force_save_video()
            if getattr(current_detector_manager, 'cap', None) is not None:
                current_detector_manager.cap.release()
        except Exception as e:
            print(f'Error during detector cleanup: {e}')

    if current_message_transmitter:
        print('Disconnecting message transmitter...')
        try:
            current_message_transmitter.disconnect()
        except Exception as e:
            print(f'Error disconnecting transmitter: {e}')

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print('GPU cache cleared')
    except Exception:
        pass

    print('Shutdown complete.')
    sys.exit(0)

def check_gpu_memory():
    """Print CUDA memory summary; return False if free VRAM < 1GB."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(device) / 1024**3

            print(f"GPU memory:")
            print(f"  Total: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated_memory:.2f} GB")
            print(f"  Reserved: {reserved_memory:.2f} GB")
            print(f"  Free (approx): {total_memory - reserved_memory:.2f} GB")

            if total_memory - reserved_memory < 1.0:
                print("Warning: less than 1GB free VRAM; inference may fail")
                return False
            return True
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False
    return True

def main():
    global current_detector_manager, current_message_transmitter

    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C for a clean exit")

    try:
        test_device = get_device()

        if test_device == 'cuda':
            if not check_gpu_memory():
                print("Try lower --resolution or CPU mode")
                use_cpu = input("Force CPU mode? (y/n): ").lower() == 'y'
                if use_cpu:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    print("Using CPU")

        parser = argparse.ArgumentParser(description='Basketball Shot and Pose Detection System')
        parser.add_argument('--game_id', type=str, default="test_game",
                          help='Game ID for messaging')
        parser.add_argument('--resolution', type=int, default=1280,
                           help='YOLO input size (default 1280)')
        parser.add_argument('--rtsp_url', type=str,
                           help='RTSP URL (e.g. rtsp://user:pass@host:port/path)')
        parser.add_argument('--json', type=str,
                           help='Launch from configs/results/<name>.json (name without .json)')

        parser.add_argument('video_path', nargs='?', type=str,
                           help='Input video path')
        parser.add_argument('start_end_lines_path', nargs='?', type=str, default='',
                           help='Start/end lines JSON path')
        parser.add_argument('three_point_lines_path', nargs='?', type=str, default='',
                           help='Three-point area JSON path')
        parser.add_argument('fps', nargs='?', type=float, default=60,
                           help='FPS for timing (default 60)')
        parser.add_argument('gender', nargs='?', type=str, default='M',
                           help='M or F (default M)')

        args = parser.parse_args()

        os.makedirs('outputs', exist_ok=True)
        os.makedirs('weights', exist_ok=True)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('configs/results', exist_ok=True)

        required_models = ['Basketball.pt' , 'Pose.pt']

        for model in required_models:
            if not os.path.exists(f'weights/{model}'):
                print(f"Error: Model file 'weights/{model}' not found")
                return

        json_config_path = None
        json_data = None
        if args.json:
            json_config_path = f"configs/results/{args.json}.json"
            if not os.path.exists(json_config_path):
                print(f"Error: JSON config not found: {json_config_path}")
                return

            try:
                import json as json_module
                with open(json_config_path, 'r', encoding='utf-8') as f:
                    json_data = json_module.load(f)

                if 'fileName' not in json_data:
                    print("Error: JSON missing 'fileName'")
                    return
                if 'courtName' not in json_data:
                    print("Error: JSON missing 'courtName'")
                    return
                if 'frameRate' not in json_data:
                    print("Error: JSON missing 'frameRate'")
                    return

                video_filename = json_data['fileName']
                court_name = json_data['courtName']
                json_fps = json_data.get('frameRate', 60)

                video_path = f"inputs/{video_filename}"
                if not os.path.exists(video_path):
                    print(f"Error: video not found: {video_path}")
                    return

                video_base_name = os.path.splitext(os.path.basename(video_path))[0]
                if isinstance(court_name, str) and court_name.strip() == "":
                    start_end_config_path = f"configs/start_end_lines/{video_base_name}_start_end_lines.json"
                    three_point_config_path = f"configs/three_point_areas/{video_base_name}_three_point.json"
                else:
                    start_end_config_path = f"configs/start_end_lines/{court_name}_start_end_lines.json"
                    three_point_config_path = f"configs/three_point_areas/{court_name}_three_point.json"

                args.video_path = video_path
                args.start_end_lines_path = start_end_config_path
                args.three_point_lines_path = three_point_config_path
                args.fps = json_fps
                args.gender = 'M'

                print("\n=== JSON launch ===")
                print(f"Config: {json_config_path}")
                print(f"Video: {video_path}")
                print(f"Court: {court_name if court_name.strip() else '(same as video base name)'}")
                print(f"FPS: {json_fps}")

            except json_module.JSONDecodeError as e:
                print(f"Error: invalid JSON: {e}")
                return
            except Exception as e:
                print(f"Error reading JSON: {e}")
                return

        use_parameter_mode = args.video_path is not None
        input_source = None

        if use_parameter_mode:
            print("\n=== CLI / parameter launch ===")

            if not os.path.exists(args.video_path):
                print(f"Error: video not found: {args.video_path}")
                return

            input_source = InputSource(args.video_path, InputType.FILE)

            video_base_name = os.path.splitext(os.path.basename(args.video_path))[0]

            start_end_config_path = args.start_end_lines_path
            if not start_end_config_path:
                default_path = f"configs/start_end_lines/{video_base_name}_start_end_lines.json"
                if os.path.exists(default_path):
                    start_end_config_path = default_path
                else:
                    start_end_config_path = None
            elif not os.path.exists(start_end_config_path):
                pass

            three_point_config_path = args.three_point_lines_path
            if not three_point_config_path:
                default_path = f"configs/three_point_areas/{video_base_name}_three_point.json"
                if os.path.exists(default_path):
                    three_point_config_path = default_path
                else:
                    three_point_config_path = None
            elif not os.path.exists(three_point_config_path):
                pass

            print(f"Video: {args.video_path}")
            print(f"Start/end config: {start_end_config_path if start_end_config_path else '(none; will mark)'}")
            print(f"Three-point config: {three_point_config_path if three_point_config_path else '(none; will mark)'}")
            print(f"FPS: {args.fps}")
            print(f"Gender: {args.gender}")

        else:
            input_source = get_video_source(args)
            if not input_source:
                print("No input source selected. Exiting.")
                return

        print(f"\nProcessing {input_source}...")
        print(f"YOLO resolution: {args.resolution}x{args.resolution}")
        print(f"Device: {test_device}")

        message_transmitter = MessageTransmitter()
        current_message_transmitter = message_transmitter

        try:
            if not message_transmitter.connect():
                print("Warning: message server unavailable; events will not be sent")
                message_transmitter = None
                current_message_transmitter = None
            else:
                print("Connected to message server")
        except Exception as e:
            print(f"Message server connection error: {e}")
            message_transmitter = None
            current_message_transmitter = None

        print("Initializing detector...")
        try:
            if input_source.source_type in [InputType.FILE, InputType.RTSP]:
                detector_manager = DetectorManager(
                    input_source.source_path,
                    message_transmitter=message_transmitter,
                    model_resolution=args.resolution
                )
            elif input_source.source_type == InputType.CAMERA:
                detector_manager = DetectorManager(
                    int(input_source.source_path),
                    message_transmitter=message_transmitter,
                    model_resolution=args.resolution
                )
            else:
                print(f"Input type {input_source.source_type} not supported")
                return

            current_detector_manager = detector_manager

            if use_parameter_mode:
                video_base_name = os.path.splitext(os.path.basename(args.video_path))[0]

                start_end_loaded = False
                if start_end_config_path and os.path.exists(start_end_config_path):
                    try:
                        with open(start_end_config_path, 'r') as f:
                            import json
                            config = json.load(f)
                            line_points = config.get("line_points", [])
                            if line_points:
                                detector_manager.start_end_line_marker.line_points = line_points
                                print(f"Start/end lines loaded: {start_end_config_path}")
                                start_end_loaded = True
                    except Exception as e:
                        print(f"Failed to load start/end config: {e}")
                        start_end_loaded = False

                three_point_loaded = False
                if three_point_config_path and os.path.exists(three_point_config_path):
                    try:
                        with open(three_point_config_path, 'r') as f:
                            import json
                            config = json.load(f)
                            arc_points = config.get("arc_points", [])
                            if len(arc_points) >= 3:
                                detector_manager.shot_detector.three_point_marker.arc_points = arc_points
                                print(f"Three-point area loaded: {three_point_config_path}")
                                three_point_loaded = True
                    except Exception as e:
                        print(f"Failed to load three-point config: {e}")
                        three_point_loaded = False

                detector_manager.start_end_config_path = start_end_config_path
                detector_manager.three_point_config_path = three_point_config_path
                detector_manager.video_base_name = video_base_name
                detector_manager.use_parameter_mode = True
                detector_manager.parameter_mode_fps = args.fps

                if not start_end_loaded or len(detector_manager.start_end_line_marker.line_points) == 0:
                    detector_manager.need_start_end_marking = True
                    if json_config_path:
                        print("Start/end lines missing or empty; will auto-mark")
                else:
                    detector_manager.need_start_end_marking = False

                if not three_point_loaded or len(detector_manager.shot_detector.three_point_marker.arc_points) < 3:
                    detector_manager.need_three_point_marking = True
                    if json_config_path:
                        print("Three-point area missing or empty; will auto-mark")
                else:
                    detector_manager.need_three_point_marking = False

                if args.fps > 0:
                    detector_manager.video_fps = args.fps
                    print(f"Using FPS: {args.fps}")

                if args.gender:
                    detector_manager.gender = args.gender.upper()
                    print(f"Gender: {args.gender}")

        except Exception as e:
            print(f"Detector init failed: {e}")
            print(traceback.format_exc())

            if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
                print("\nTry:")
                print("  1. Lower resolution: --resolution 320")
                print("  2. CPU: CUDA_VISIBLE_DEVICES=-1")
                print("  3. Close other GPU apps")
            return

        print("Detector ready")

        try:
            output_path = detector_manager.run()

            if output_path:
                print(f"\nDone. Output: {output_path}")
        except Exception as e:
            print(f"Runtime error: {e}")
            print(traceback.format_exc())

            if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
                print("\nLikely GPU OOM. Try lower --resolution or CPU mode.")

        finally:
            if json_config_path and json_data:
                try:
                    has_valid_score = False
                    final_score = 0
                    if current_detector_manager and hasattr(current_detector_manager, 'state_machine'):
                        if current_detector_manager.state_machine.is_complete():
                            has_valid_score = True
                            if hasattr(current_detector_manager, 'exam_score') and current_detector_manager.exam_score is not None:
                                final_score = current_detector_manager.exam_score

                    if has_valid_score:
                        json_data['totalScore'] = final_score
                        print(f"\nWrote totalScore to JSON: {final_score}")
                    else:
                        print("\nTest not complete; totalScore unchanged")

                    if current_message_transmitter:
                        events = current_message_transmitter.get_events()
                        json_data['events'] = events
                        print(f"Wrote {len(events)} events to JSON")

                    import json as json_module
                    with open(json_config_path, 'w', encoding='utf-8') as f:
                        json_module.dump(json_data, f, indent=4, ensure_ascii=False)
                    print(f"Updated: {json_config_path}")

                except Exception as json_error:
                    print(f"JSON write error: {json_error}")

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("GPU cache cleared")
            except Exception:
                pass

    except Exception as e:
        print(f"Fatal error: {e}")
        print(traceback.format_exc())
    finally:
        if current_message_transmitter:
            current_message_transmitter.disconnect()

        current_detector_manager = None
        current_message_transmitter = None

if __name__ == "__main__":
    main()
