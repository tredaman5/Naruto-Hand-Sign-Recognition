import numpy as np
from src.features.normalize import normalize_landmarks


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """
    Convert MediaPipe hand landmarks into a NumPy array.

    Returns:
        shape (21, 3)
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def extract_features(hand_landmarks) -> np.ndarray:
    """
    Convert MediaPipe landmarks into normalized ML features.

    Returns:
        shape (63,)
    """
    landmark_array = landmarks_to_array(hand_landmarks)
    return normalize_landmarks(landmark_array)


def create_feature_names() -> list[str]:
    """
    Creates column names for saved CSV files.
    """
    names = []

    for i in range(21):
        names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

    return names