"""
Real-time AVSR demo. Run: python demo.py [--mode audio|video|fusion] [--norm_stats PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from avsr.inference import run


def _parse():
    parser = argparse.ArgumentParser(description="Real-time AVSR demo — press SPACE to speak.")
    parser.add_argument(
        "--mode",
        choices=["audio", "video", "fusion"],
        default=None,
        help="Model to run (default: value in config.yaml).",
    )
    parser.add_argument(
        "--norm_stats",
        default=None,
        metavar="PATH",
        help="Path to dataset_train.npz for video Z-score normalisation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(mode=args.mode, norm_stats=args.norm_stats)
