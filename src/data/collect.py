import cv2
import mediapipe as mp

from src.data.save_sample import save_landmark_sample
from src.features.extract import extract_features
from src.labels.naruto_signs import NARUTO_SIGNS


def collect_samples(label: str, camera_index: int = 0) -> None:
    if label not in NARUTO_SIGNS:
        raise ValueError(f"Invalid label '{label}'. Choose from: {NARUTO_SIGNS}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    saved_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_landmarks = None
            handedness = "Unknown"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

            cv2.putText(
                frame,
                f"Label: {label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Saved: {saved_count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Press S = save | Q = quit",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Naruto Hand Sign Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if hand_landmarks is None:
                    print("No hand detected. Sample not saved.")
                else:
                    features = extract_features(hand_landmarks)
                    save_landmark_sample(label, handedness, features)
                    saved_count += 1
                    print(f"Saved {label} sample #{saved_count}")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()