from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT_DIR.parent
sys.path.insert(0, str(TRAIN_ROOT))
sys.path.insert(0, str(ROOT_DIR))

from src.cli.demo_inference import main


if __name__ == "__main__":
    main()
