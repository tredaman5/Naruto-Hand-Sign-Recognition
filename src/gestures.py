from src.utils import distance, get_landmark_points


THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

WRIST = 0


def is_finger_up(points, tip_idx: int, pip_idx: int) -> bool:
    return points[tip_idx][1] < points[pip_idx][1]


def classify_gesture(hand_landmarks) -> str:
    points = get_landmark_points(hand_landmarks)

    index_up = is_finger_up(points, INDEX_TIP, INDEX_PIP)
    middle_up = is_finger_up(points, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = is_finger_up(points, RING_TIP, RING_PIP)
    pinky_up = is_finger_up(points, PINKY_TIP, PINKY_PIP)

    thumb_tip = points[THUMB_TIP]
    wrist = points[WRIST]
    index_tip = points[INDEX_TIP]
    middle_tip = points[MIDDLE_TIP]

    thumb_far_from_wrist = distance(thumb_tip, wrist) > 0.2
    index_middle_close = distance(index_tip, middle_tip) < 0.08

    if index_up and middle_up and ring_up and pinky_up and thumb_far_from_wrist:
        return "Open Palm"

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "Fist"

    if index_up and middle_up and not ring_up and not pinky_up and not index_middle_close:
        return "Peace Sign"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Pointing"

    if thumb_far_from_wrist and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Up"

    return "Unknown"