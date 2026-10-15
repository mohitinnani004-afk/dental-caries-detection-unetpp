"""
Root-level convenience launcher for model training.
Can be executed from any directory:
    python train.py --epochs 40 --batch_size 8
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import main

if __name__ == "__main__":
    main()
