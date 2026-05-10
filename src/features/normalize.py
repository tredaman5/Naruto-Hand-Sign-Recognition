import numpy as np


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    landmarks = np.array(landmarks, dtype=np.float32)

    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[0]
    normalized = landmarks - wrist

    scale = np.max(np.linalg.norm(normalized, axis=1))

    if scale < 1e-6:
        return normalized.flatten()

    return (normalized / scale).flatten()