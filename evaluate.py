"""
Root-level convenience launcher for model evaluation.
Can be executed from any directory:
    python evaluate.py
"""

from pathlib import Path
import sys

# Automatically set sys.path to project root regardless of execution cwd
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import main

if __name__ == "__main__":
    main()
