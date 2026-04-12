"""
Real-time AVSR demo. Run: python demo.py [--mode audio|video|fusion] [--norm_stats PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from avsr.inference import list_devices, run


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
    parser.add_argument(
        "--video-device",
        type=int,
        default=0,
        metavar="INDEX",
        help="Camera device index (default: 0). Use --list-devices to see options.",
    )
    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Audio input device index (default: system default). Use --list-devices to see options.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio and video devices, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.list_devices:
        list_devices()
        sys.exit(0)
    run(
        mode=args.mode,
        norm_stats=args.norm_stats,
        video_device=args.video_device,
        audio_device=args.audio_device,
    )
