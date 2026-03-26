from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
import json
import os

class State(Enum):
    INIT = auto()
    FIRST_SHOT = auto()
    RETRY_SHOT = auto()
    OUT_3PT = auto()
    COMPLETE = auto()

class Event(Enum):
    FIRST_SHOT_MADE = "first_shot_made"
    FIRST_SHOT_MISSED = "first_shot_missed"
    RETRY_SHOT_MADE = "retry_shot_made"
    RETRY_SHOT_MISSED = "retry_shot_missed"
    OUT_3PT = "out_three_point"
    START = "start"
    END = "end"

class StateMachine:
    """Layup test FSM (no foul handling)."""
    
    def __init__(self):
        self.state = State.INIT
        self.valid_shots = 0
        self.retry_count = 0
        self.can_start_new = True
        
        self.state_history: List[Dict] = []
        
    def get_state_name(self) -> str:
        state_names = {
            State.INIT: "Initial",
            State.FIRST_SHOT: "First Shot",
            State.RETRY_SHOT: "Retry Shot",
            State.OUT_3PT: "Out of Three-Point Line",
            State.COMPLETE: "Complete"
        }
        return state_names.get(self.state, "Unknown")
    
    def get_progress(self) -> Dict:
        return {
            "state": self.get_state_name(),
            "valid_shots": f"{self.valid_shots}/4",
            "retry_count": f"{self.retry_count}/2",
            "can_start_new": self.can_start_new,
            "is_complete": self.state == State.COMPLETE
        }
    
    def transition(self, event: Event, timestamp: float) -> Tuple[bool, Optional[str]]:
        """Apply ``event`` at ``timestamp``. Returns (ok, error_message)."""
        old_state = self.state
        error_msg = None
        
        if event == Event.END:
            is_normal_end = (self.state == State.OUT_3PT and self.valid_shots >= 4)
            self.state = State.COMPLETE
            
            if not is_normal_end:
                error_msg = (
                    f"Premature END event in state {old_state.name}" if self.state != State.OUT_3PT
                    else f"Test ended with insufficient valid shots ({self.valid_shots}/4)"
                )
            
            self.state_history.append({
                "timestamp": timestamp,
                "event": event.value,
                "from_state": old_state.name,
                "to_state": self.state.name,
                "valid_shots": self.valid_shots,
                "retry_count": self.retry_count,
                "can_start_new": self.can_start_new,
                "is_normal_end": is_normal_end,
                "error": error_msg
            })
            return True, error_msg
        
        if self.state == State.COMPLETE:
            return False, "Test already completed"
        
        if self.state == State.INIT:
            if event == Event.START:
                self.state = State.FIRST_SHOT
                self.can_start_new = True
            else:
                error_msg = f"Invalid event {event.value} in INIT state"
                return False, error_msg
        
        elif self.state == State.FIRST_SHOT:
            if not self.can_start_new:
                error_msg = "Need to exit three-point line first"
                return False, error_msg
                
            if event == Event.FIRST_SHOT_MADE:
                self.valid_shots += 1
                self.retry_count = 0
                self.can_start_new = False
                self.state = State.OUT_3PT
            elif event == Event.FIRST_SHOT_MISSED:
                self.retry_count = 0
                self.state = State.RETRY_SHOT
            else:
                error_msg = f"Invalid event {event.value} in FIRST_SHOT state"
                return False, error_msg
        
        elif self.state == State.RETRY_SHOT:
            if event == Event.RETRY_SHOT_MADE:
                self.valid_shots += 1
                self.retry_count = 0
                self.can_start_new = False
                self.state = State.OUT_3PT
            elif event == Event.RETRY_SHOT_MISSED:
                self.retry_count += 1
                if self.retry_count >= 2:
                    self.can_start_new = False
                    self.state = State.OUT_3PT
            else:
                error_msg = f"Invalid event {event.value} in RETRY_SHOT state"
                return False, error_msg
        
        elif self.state == State.OUT_3PT:
            if event == Event.OUT_3PT:
                if self.valid_shots >= 4:
                    self.can_start_new = False
                else:
                    self.can_start_new = True
                    self.state = State.FIRST_SHOT
            else:
                error_msg = f"Invalid event {event.value} in OUT_3PT state"
                return False, error_msg
        
        if old_state != self.state:
            self.state_history.append({
                "timestamp": timestamp,
                "event": event.value,
                "from_state": old_state.name,
                "to_state": self.state.name,
                "valid_shots": self.valid_shots,
                "retry_count": self.retry_count,
                "can_start_new": self.can_start_new,
                "error": error_msg
            })
            return True, error_msg
            
        return False, "No state change occurred"
    
    def is_complete(self) -> bool:
        return (self.state == State.COMPLETE and 
                self.valid_shots >= 4)
    
    def get_next_requirement(self) -> str:
        if self.state == State.INIT:
            return "Ready to start"
        elif self.state == State.FIRST_SHOT:
            if not self.can_start_new:
                return "Need to exit three-point line first"
            return "Attempt first shot"
        elif self.state == State.RETRY_SHOT:
            return f"Retry shot ({self.retry_count}/2)"
        elif self.state == State.OUT_3PT:
            if self.valid_shots >= 4:
                return "Exit three-point line to complete"
            return "Exit three-point line for next attempt"
        else:
            return "Test complete"
    
    def get_completion_info(self) -> Tuple[bool, str]:
        """(success, message) after END; uses last END record in history when present."""
        if self.state != State.COMPLETE:
            return False, "Test not completed"
        
        for record in reversed(self.state_history):
            if record.get("event") == "end":
                is_normal_end = record.get("is_normal_end", False)
                error_msg = record.get("error")
                valid_shots = record.get("valid_shots", 0)
                
                if is_normal_end:
                    return True, "Test Complete Successfully"
                else:
                    error_info = f"Error: {error_msg}" if error_msg else "Unknown error"
                    return False, f"{error_info} (Valid shots: {valid_shots}/4)"
        
        if self.valid_shots >= 4:
            return True, "Test Complete Successfully"
        else:
            return False, f"Test ended with insufficient valid shots ({self.valid_shots}/4)"

if __name__ == "__main__":
    def run_test_sequence(name: str, sequence: List[Tuple[Event, float]]):
        print(f"\n=== Case: {name} ===")
        state_machine = StateMachine()
        
        for event, timestamp in sequence:
            print(f"\nState: {state_machine.get_state_name()}")
            print(f"Event: {event.value} @ {timestamp}s")
            success, error_msg = state_machine.transition(event, timestamp)
            progress = state_machine.get_progress()
            print(f"Transition: {'ok' if success else 'failed'}")
            print(f"Progress: {progress}")
            if not success:
                print(f"Reason: {error_msg}")
            
        print(f"\nFinal state: {state_machine.get_state_name()}")
        print(f"Valid shots: {state_machine.valid_shots}/4")
        print(f"History:")
        for record in state_machine.state_history:
            print(f"- {record.get('from_state')} -> {record.get('to_state')}")
            if record.get('error'):
                print(f"  error: {record['error']}")
            if 'is_normal_end' in record:
                print(f"  normal_end: {record['is_normal_end']}")
    
    normal_sequence = [
        (Event.START, 0),
        (Event.FIRST_SHOT_MADE, 1),
        (Event.OUT_3PT, 2),
        (Event.FIRST_SHOT_MADE, 3),
        (Event.OUT_3PT, 4),
        (Event.FIRST_SHOT_MADE, 5),
        (Event.OUT_3PT, 6),
        (Event.FIRST_SHOT_MADE, 7),
        (Event.OUT_3PT, 8),
        (Event.END, 9),
    ]
    run_test_sequence("full success (4 makes)", normal_sequence)
    
    retry_sequence = [
        (Event.START, 0),
        (Event.FIRST_SHOT_MISSED, 1),
        (Event.RETRY_SHOT_MADE, 2),
        (Event.OUT_3PT, 3),
        (Event.FIRST_SHOT_MADE, 4),
        (Event.OUT_3PT, 5),
        (Event.FIRST_SHOT_MISSED, 6),
        (Event.RETRY_SHOT_MISSED, 7),
        (Event.RETRY_SHOT_MADE, 8),
        (Event.OUT_3PT, 9),
        (Event.FIRST_SHOT_MADE, 10),
        (Event.OUT_3PT, 11),
        (Event.END, 12),
    ]
    run_test_sequence("with retry shots", retry_sequence)
    
    early_end_sequence = [
        (Event.START, 0),
        (Event.FIRST_SHOT_MADE, 1),
        (Event.OUT_3PT, 2),
        (Event.END, 3),
    ]
    run_test_sequence("early END (not enough makes)", early_end_sequence)
    
    invalid_sequence = [
        (Event.START, 0),
        (Event.FIRST_SHOT_MADE, 1),
        (Event.FIRST_SHOT_MADE, 2),
        (Event.RETRY_SHOT_MADE, 3),
        (Event.END, 4),
    ]
    run_test_sequence("invalid transitions", invalid_sequence)
    
    failed_retry_sequence = [
        (Event.START, 0),
        (Event.FIRST_SHOT_MISSED, 1),
        (Event.RETRY_SHOT_MISSED, 2),
        (Event.RETRY_SHOT_MISSED, 3),
        (Event.OUT_3PT, 4),
        (Event.END, 5),
    ]
    run_test_sequence("two retry misses", failed_retry_sequence)
    
    wrong_state_end_sequence = [
        (Event.START, 0),
        (Event.FIRST_SHOT_MADE, 1),
        (Event.OUT_3PT, 2),
        (Event.FIRST_SHOT_MISSED, 3),
        (Event.END, 4),
    ]
    run_test_sequence("END during RETRY_SHOT", wrong_state_end_sequence)
