"""Data handling utilities for LayoutLMv3 pipeline.

- copy_dataset: sync images and CSV from Modal volume to workspace.
- convert_csv_to_jsonl: invoke the existing conversion script.
- load_dataset: return a ``datasets.DatasetDict`` for a given split.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Tuple


def copy_dataset(cfg_split) -> None:
    """Copy images and CSV for the provided split configuration.

    Args:
        cfg_split: a ``SplitConfig`` dataclass instance containing
            ``images_dir``, ``csv_path`` and ``output_dir``.
    """
    # Ensure output directory exists
    cfg_split.output_dir.mkdir(parents=True, exist_ok=True)
    # Copy images directory
    shutil.copytree(cfg_split.images_dir, cfg_split.output_dir / "imgs", dirs_exist_ok=True)
    # Copy CSV file
    shutil.copy2(cfg_split.csv_path, cfg_split.output_dir / f"{cfg_split.output_dir.name}_df.csv")
    print(f"✅ {cfg_split.output_dir.name.capitalize()} dataset copied to {cfg_split.output_dir}")


def convert_csv_to_jsonl(csv_path: Path, jsonl_dir: Path) -> None:
    """Run the conversion script located in ``layoutlmv3/scripts``.

    The script expects ``--csv`` and ``--out`` arguments.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "convert_csv_to_jsonl.py"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python",
        str(script_path),
        "--csv",
        str(csv_path),
        "--out",
        str(jsonl_dir),
    ])
    print(f"✅ CSV → JSONL conversion completed: {jsonl_dir}")


def load_dataset(jsonl_dir: Path, split_name: str = "train"):
    """Load a DatasetDict from the JSONL file produced by the conversion step.

    Returns a ``datasets.DatasetDict`` with a single split.
    """
    from datasets import load_dataset
    jsonl_path = jsonl_dir / f"{split_name}_df.jsonl"
    return load_dataset("json", data_files={split_name: str(jsonl_path)})
