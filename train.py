"""Train all AVSR models. Run: python train.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from avsr.train import run

if __name__ == "__main__":
    run()
