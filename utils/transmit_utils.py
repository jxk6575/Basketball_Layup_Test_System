from typing import Dict, Any

class MessageTransmitter:
    def __init__(self):
        self.events = []

    def connect(self) -> bool:
        """Stub: ActiveMQ removed; API kept for callers."""
        return True

    def disconnect(self):
        """Stub disconnect."""
        pass

    def send_message(self, message: Dict[str, Any]) -> bool:
        """Stub send; always True."""
        return True
    
    def send_error(self, error_message: str) -> bool:
        """Stub error path; always True."""
        return True

    def send_event(self, video_name: str, time_stamp: float, event_type: str, frame_number: int = None) -> bool:
        """
        Record an event when ``frame_number`` is set: {id, timestamp: frame, type}.

        ``video_name`` / ``time_stamp`` kept for API compatibility (not stored).
        ``event_type``: first_shot_made | first_shot_missed | retry_shot_made |
        retry_shot_missed | out_three_point | start | end
        """
        if frame_number is not None:
            event_record = {
                'id': len(self.events) + 1,
                'timestamp': frame_number,
                'type': event_type
            }
            self.events.append(event_record)
        
        return True
    
    def get_events(self):
        return self.events.copy()
    
    def clear_events(self):
        self.events = []
    
    def send_first_shot_made(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'first_shot_made', frame_number)
    
    def send_first_shot_missed(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'first_shot_missed', frame_number)
    
    def send_retry_shot_made(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'retry_shot_made', frame_number)
        
    def send_retry_shot_missed(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'retry_shot_missed', frame_number)
    
    def send_three_point_exit(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'out_three_point', frame_number)

    def send_timing_start(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'start', frame_number)

    def send_timing_end(self, video_name: str, time_stamp: float, frame_number: int = None) -> bool:
        return self.send_event(video_name, time_stamp, 'end', frame_number)

def main():
    transmitter = MessageTransmitter()
    transmitter.connect()
    transmitter.send_first_shot_made('test_video.mp4', 0.0, 100)
    transmitter.send_three_point_exit('test_video.mp4', 0.0, 200)
    transmitter.send_timing_end('test_video.mp4', 0.0, 300)
    transmitter.send_timing_start('test_video.mp4', 0.0, 400)
    transmitter.send_first_shot_missed('test_video.mp4', 0.0, 500)
    transmitter.send_retry_shot_made('test_video.mp4', 0.0, 600)
    transmitter.send_retry_shot_missed('test_video.mp4', 0.0, 700)
    
    events = transmitter.get_events()
    print(f"Recorded {len(events)} events:")
    for event in events:
        print(f"  {event}")
    transmitter.disconnect()

if __name__ == '__main__':
    main()
