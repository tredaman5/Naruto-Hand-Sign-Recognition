import cv2

from src.camera import Camera
from src.detector import HandDetector
from src.gestures import classify_gesture
from src.automation import ActionController
from src.utils import FPSCounter, draw_text
from src.config import (
    CAMERA_INDEX,
    WINDOW_NAME,
    MODEL_COMPLEXITY,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    DISPLAY_FPS,
    DISPLAY_GESTURE,
    DISPLAY_LANDMARKS,
    ACTION_COOLDOWN_SECONDS,
)


def run_app() -> None:
    camera = Camera(CAMERA_INDEX)
    fps_counter = FPSCounter()
    action_controller = ActionController(ACTION_COOLDOWN_SECONDS)

    detector = HandDetector(
        model_complexity=MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    if not camera.is_opened():
        print("Error: Could not open webcam.")
        return

    try:
        while camera.is_opened():
            success, frame = camera.read_frame()

            if not success:
                print("Empty frame.")
                continue

            frame = cv2.flip(frame, 1)
            results = detector.process(frame)

            gesture_label = "No Hands Detected"
            action_message = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if DISPLAY_LANDMARKS:
                        detector.draw_landmarks(frame, hand_landmarks)

                    gesture_label = classify_gesture(hand_landmarks)
                    action_message = action_controller.handle_action(gesture_label)

            if DISPLAY_GESTURE:
                draw_text(frame, f"Gesture: {gesture_label}", 10, 30)

            if action_message:
                draw_text(frame, f"Action: {action_message}", 10, 65)

            if DISPLAY_FPS:
                fps = fps_counter.update()
                draw_text(frame, f"FPS: {fps:.2f}", 10, 100)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()