import time


class ActionController:
    def __init__(self, cooldown_seconds: float = 1.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_action_time = 0.0
        self.last_gesture = None

    def can_trigger(self, gesture: str) -> bool:
        current_time = time.time()

        if gesture == "Unknown":
            return False

        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.last_action_time = current_time
            return True

        if current_time - self.last_action_time >= self.cooldown_seconds:
            self.last_action_time = current_time
            return True

        return False

    def handle_action(self, gesture: str) -> str:
        if not self.can_trigger(gesture):
            return ""

        action_map = {
            "Open Palm": "System Idle",
            "Fist": "Pause Command Triggered",
            "Peace Sign": "Screenshot Command Triggered",
            "Pointing": "Pointer Command Triggered",
            "Thumbs Up": "Volume Up Command Triggered",
        }

        action_message = action_map.get(gesture, "")
        if action_message:
            print(action_message)

        return action_message