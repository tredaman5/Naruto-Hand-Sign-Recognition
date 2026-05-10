import numpy as np

from src.features.normalize import normalize_landmarks


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def create_feature_names() -> list[str]:
    names = []

    for hand in ["left", "right"]:
        for i in range(21):
            names.extend([
                f"{hand}_lm_{i}_x",
                f"{hand}_lm_{i}_y",
                f"{hand}_lm_{i}_z",
            ])

    return names


def extract_two_hand_features(results) -> np.ndarray | None:
    """
    Extract normalized two-hand features from MediaPipe results.

    Returns:
        np.ndarray of shape (126,) if both hands are detected
        None if fewer than 2 hands are detected
    """
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None

    hands = {}

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness,
    ):
        label = handedness.classification[0].label.lower()

        if label in ["left", "right"]:
            hands[label] = hand_landmarks

    if "left" not in hands or "right" not in hands:
        return None

    left_array = landmarks_to_array(hands["left"])
    right_array = landmarks_to_array(hands["right"])

    left_features = normalize_landmarks(left_array)
    right_features = normalize_landmarks(right_array)

    return np.concatenate([left_features, right_features])