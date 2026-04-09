"""Build train/val/test NPZ datasets from raw recordings. Run: python preprocess.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from avsr.preprocessing import build_dataset

if __name__ == "__main__":
    build_dataset()
