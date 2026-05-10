import csv
from pathlib import Path

import numpy as np
import pytest

from src.data.save_sample import (
    create_dataset_header,
    ensure_dataset_exists,
    save_landmark_sample,
)


def test_create_dataset_header_length():
    header = create_dataset_header()

    assert len(header) == 65
    assert header[0] == "label"
    assert header[1] == "handedness"
    assert header[2] == "lm_0_x"
    assert header[-1] == "lm_20_z"


def test_ensure_dataset_exists_creates_file(tmp_path):
    dataset_path = tmp_path / "landmarks.csv"

    ensure_dataset_exists(dataset_path)

    assert dataset_path.exists()

    with dataset_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

    assert header == create_dataset_header()


def test_save_landmark_sample_adds_row(tmp_path):
    dataset_path = tmp_path / "landmarks.csv"
    features = np.random.rand(63)

    save_landmark_sample(
        label="tiger",
        handedness="Right",
        features=features,
        dataset_path=dataset_path,
    )

    with dataset_path.open("r", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 2
    assert rows[0] == create_dataset_header()
    assert rows[1][0] == "tiger"
    assert rows[1][1] == "Right"
    assert len(rows[1]) == 65


def test_save_landmark_sample_rejects_bad_feature_length(tmp_path):
    dataset_path = tmp_path / "landmarks.csv"
    bad_features = np.random.rand(20)

    with pytest.raises(ValueError):
        save_landmark_sample(
            label="tiger",
            handedness="Right",
            features=bad_features,
            dataset_path=dataset_path,
        )