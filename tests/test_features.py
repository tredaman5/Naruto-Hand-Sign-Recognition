import numpy as np
import pytest

from src.features.normalize import normalize_landmarks
from src.features.extract import create_feature_names
from src.labels.naruto_signs import NARUTO_SIGNS


def test_naruto_signs_exist():
    assert "tiger" in NARUTO_SIGNS
    assert "serpent" in NARUTO_SIGNS
    assert len(NARUTO_SIGNS) == 6


def test_create_feature_names_length():
    names = create_feature_names()

    assert len(names) == 63
    assert names[0] == "lm_0_x"
    assert names[1] == "lm_0_y"
    assert names[2] == "lm_0_z"
    assert names[-1] == "lm_20_z"


def test_normalize_landmarks_output_shape():
    landmarks = np.random.rand(21, 3)

    features = normalize_landmarks(landmarks)

    assert features.shape == (63,)


def test_normalize_landmarks_bad_shape_raises_error():
    bad_landmarks = np.random.rand(10, 3)

    with pytest.raises(ValueError):
        normalize_landmarks(bad_landmarks)