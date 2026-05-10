import argparse

from src.data.collect import collect_samples
from src.labels.naruto_signs import NARUTO_SIGNS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect MediaPipe landmark samples for Naruto hand signs."
    )

    parser.add_argument(
        "--label",
        required=True,
        choices=NARUTO_SIGNS,
        help="Naruto hand sign label to collect.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index. Default is 0.",
    )

    args = parser.parse_args()

    collect_samples(label=args.label, camera_index=args.camera)


if __name__ == "__main__":
    main()