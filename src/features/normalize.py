import numpy as np


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks so the model learns hand shape,
    not hand location or distance from camera.

    Expected shape:
        (21, 3)

    Returns:
        flattened vector of shape (63,)
    """
    landmarks = np.array(landmarks, dtype=np.float32)

    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[0]
    normalized = landmarks - wrist

    scale = np.linalg.norm(normalized[9])

    if scale < 1e-6:
        scale = np.max(np.linalg.norm(normalized, axis=1))

    if scale < 1e-6:
        return normalized.flatten()

    normalized = normalized / scale

    return normalized.flatten()