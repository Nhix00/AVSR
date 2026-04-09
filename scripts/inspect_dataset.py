"""
Inspect train/val/test NPZ datasets: shapes, class balance, augmentation distribution.

Run: python scripts/inspect_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from collections import Counter

from avsr.config import cfg


def _reverse_class_map() -> dict:
    seen = {}
    for word, idx in cfg.dataset.class_map.items():
        if idx not in seen:
            seen[idx] = word
    return seen


def inspect(file_path: str) -> None:
    """Print shape, class distribution, and augmentation breakdown for an NPZ file."""
    print(f"\n{'='*45}")
    print(f"  {file_path}")
    print(f"{'='*45}")

    if not Path(file_path).exists():
        print(f"  File not found: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    X_audio = data["X_audio"]
    X_video = data["X_video"]
    Y_labels = data["Y_labels"]
    Y_metadata = data["Y_metadata"]

    n = len(Y_labels)
    print(f"  Total samples : {n}")
    print(f"  X_audio shape : {X_audio.shape}")
    print(f"  X_video shape : {X_video.shape}")

    id_to_class = _reverse_class_map()
    print("\n  Class distribution:")
    for label, count in sorted(Counter(Y_labels).items()):
        name = id_to_class.get(label, str(label))
        print(f"    [{label:>2}] {name:<10}  {count:>4}  ({count/n*100:.1f}%)")

    print("\n  Augmentation conditions:")
    for tag, count in sorted(Counter(Y_metadata).items()):
        print(f"    {str(tag):<22}  {count:>4}  ({count/n*100:.1f}%)")


if __name__ == "__main__":
    for split in [cfg.dataset.train_file, cfg.dataset.val_file, cfg.dataset.test_file]:
        inspect(split)
