import time

import cv2
import mediapipe as mp

from src.data.save_sample import save_landmark_sample
from src.features.extract import extract_two_hand_features
from src.labels.naruto_signs import NARUTO_SIGNS


def collect_samples(
    label: str,
    camera_index: int = 0,
    capture_interval: float = 1.0,
) -> None:
    if label not in NARUTO_SIGNS:
        raise ValueError(f"Invalid label '{label}'. Choose from: {NARUTO_SIGNS}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    saved_count = 0
    auto_capture = False
    last_capture_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_count = 0

            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

            features = extract_two_hand_features(results)
            ready = features is not None

            now = time.time()

            if auto_capture and ready and now - last_capture_time >= capture_interval:
                save_landmark_sample(label=label, features=features)
                saved_count += 1
                last_capture_time = now
                print(f"Auto-saved {label} two-hand sample #{saved_count}")

            status_text = "READY" if ready else "NEED 2 HANDS"
            auto_text = "ON" if auto_capture else "OFF"

            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Hands detected: {hand_count}/2", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Status: {status_text}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Auto Capture: {auto_text}", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Saved: {saved_count}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, "A = auto on/off | S = save once | Q = quit", (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("Naruto Hand Sign Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):
                auto_capture = not auto_capture
                last_capture_time = 0.0
                print(f"Auto-capture {'ON' if auto_capture else 'OFF'}")

            elif key == ord("s"):
                if ready:
                    save_landmark_sample(label=label, features=features)
                    saved_count += 1
                    print(f"Saved {label} two-hand sample #{saved_count}")
                else:
                    print("Need exactly 2 hands detected. Sample not saved.")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()