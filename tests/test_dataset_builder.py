import csv

import numpy as np
import pytest

from src.data.save_sample import (
    create_dataset_header,
    ensure_dataset_exists,
    save_landmark_sample,
)


def test_create_dataset_header_length():
    header = create_dataset_header()

    assert len(header) == 127
    assert header[0] == "label"
    assert header[1] == "left_lm_0_x"
    assert header[-1] == "right_lm_20_z"


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
    features = np.random.rand(126)

    save_landmark_sample(
        label="tiger",
        features=features,
        dataset_path=dataset_path,
    )

    with dataset_path.open("r", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 2
    assert rows[0] == create_dataset_header()
    assert rows[1][0] == "tiger"
    assert len(rows[1]) == 127


def test_save_landmark_sample_rejects_bad_feature_length(tmp_path):
    dataset_path = tmp_path / "landmarks.csv"
    bad_features = np.random.rand(63)

    with pytest.raises(ValueError):
        save_landmark_sample(
            label="tiger",
            features=bad_features,
            dataset_path=dataset_path,
        )