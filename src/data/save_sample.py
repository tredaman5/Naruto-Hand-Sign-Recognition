import csv
from pathlib import Path
from typing import Iterable

from src.features.extract import create_feature_names


DEFAULT_DATASET_PATH = Path("data/processed/landmarks.csv")


def create_dataset_header() -> list[str]:
    return ["label"] + create_feature_names()


def ensure_dataset_exists(dataset_path: Path = DEFAULT_DATASET_PATH) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        with dataset_path.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(create_dataset_header())


def save_landmark_sample(
    label: str,
    features: Iterable[float],
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> None:
    features = list(features)

    if len(features) != 126:
        raise ValueError(f"Expected 126 features, got {len(features)}")

    ensure_dataset_exists(dataset_path)

    row = [label] + features

    with dataset_path.open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)