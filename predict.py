"""
Root-level convenience launcher for model prediction & lesion detection.
Can be executed from any directory:
    python predict.py --input samples/sample_proximal.png
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict import main

if __name__ == "__main__":
    main()
