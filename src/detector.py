import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        model_complexity: int = 0,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            model_complexity=model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks) -> None:
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=hand_landmarks,
            connections=self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self.hands.close()