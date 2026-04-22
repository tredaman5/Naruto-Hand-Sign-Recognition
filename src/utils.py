import math
import time
import cv2


class FPSCounter:
    def __init__(self) -> None:
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self) -> float:
        current_time = time.time()
        delta = current_time - self.prev_time

        if delta > 0:
            self.fps = 1.0 / delta

        self.prev_time = current_time
        return self.fps


def draw_text(frame, text: str, x: int, y: int) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def distance(point_a, point_b) -> float:
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def get_landmark_points(hand_landmarks):
    return [(lm.x, lm.y) for lm in hand_landmarks.landmark]