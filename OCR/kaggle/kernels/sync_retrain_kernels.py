"""Sync PICK Kaggle kernel folders + pick-train-code dataset + rebuild notebooks."""
from __future__ import annotations

import shutil
from pathlib import Path

KERNELS = Path(__file__).resolve().parent
COMMON_SRC = KERNELS.parent / "pick_kaggle_common.py"
VENDOR_ZIP = KERNELS / "vendor_pick.zip"


def _copy_common() -> None:
    for name in ("train-pick-kie", "retrain-pick-kie"):
        dst = KERNELS / name / "pick_kaggle_common.py"
        shutil.copy2(COMMON_SRC, dst)
        print(f"Copied pick_kaggle_common.py -> {name}/")
        for stale in (KERNELS / name / "vendor_pick.zip", VENDOR_ZIP):
            if stale.is_file():
                stale.unlink()
                print(f"Removed {stale.name}")


def main() -> None:
    from build_pick_kaggle_notebooks import main as rebuild

    datasets_dir = KERNELS.parent / "datasets"
    dataset_script = datasets_dir / "sync_pick_train_code_dataset.py"
    
    _copy_common()
    
    if dataset_script.is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sync_pick_train_code_dataset",
            dataset_script,
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.sync_pick_code()
        
    rebuild()
    print("Done — push pick-train-code dataset, then train-pick-kie kernel")


if __name__ == "__main__":
    main()
