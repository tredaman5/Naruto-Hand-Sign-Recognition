import csv
from pathlib import Path
from typing import Iterable

from src.features.extract import create_feature_names


DEFAULT_DATASET_PATH = Path("data/processed/landmarks.csv")


def create_dataset_header() -> list[str]:
    """
    Create the CSV header for the landmark dataset.

    Columns:
        label, handedness, lm_0_x, lm_0_y, lm_0_z, ..., lm_20_z
    """
    return ["label", "handedness"] + create_feature_names()


def ensure_dataset_exists(dataset_path: Path = DEFAULT_DATASET_PATH) -> None:
    """
    Create the dataset CSV file with headers if it does not exist.
    """
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        with dataset_path.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(create_dataset_header())


def save_landmark_sample(
    label: str,
    handedness: str,
    features: Iterable[float],
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> None:
    """
    Save a single normalized landmark feature row to the CSV dataset.

    Args:
        label: Naruto sign label, like "tiger"
        handedness: "Left" or "Right"
        features: normalized landmark vector of length 63
        dataset_path: path to CSV dataset
    """
    features = list(features)

    if len(features) != 63:
        raise ValueError(f"Expected 63 features, got {len(features)}")

    ensure_dataset_exists(dataset_path)

    row = [label, handedness] + features

    with dataset_path.open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)