"""Chỉ train NER spaCy (sau khi đã chạy ner_prepare)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
NER_MODEL_BEST = MODELS / "ner_model" / "model-best"
NER_TMP = MODELS / "ner_retrain_out"


def _patch_ner_max_steps(cfg_in: Path, cfg_out: Path, max_steps: int) -> None:
    text = cfg_in.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("max_steps ="):
            lines.append(f"max_steps = {max_steps}")
        else:
            lines.append(line)
    cfg_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    max_steps = int(os.environ.get("NER_MAX_STEPS", "6000"))
    base_cfg = TRAIN_DIR / "ner_config.cfg"
    tmp_cfg = TRAIN_DIR / "ner_config_retrain.tmp.cfg"
    _patch_ner_max_steps(base_cfg, tmp_cfg, max_steps)

    if NER_TMP.exists():
        shutil.rmtree(NER_TMP)

    cmd = [
        sys.executable,
        "-m",
        "spacy",
        "train",
        str(tmp_cfg),
        "--output",
        str(NER_TMP),
        "--paths.train",
        str(DATASETS / "ner_train.spacy"),
        "--paths.dev",
        str(DATASETS / "ner_dev.spacy"),
    ]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(TRAIN_DIR), check=True)

    new_best = NER_TMP / "model-best"
    if not new_best.exists():
        raise SystemExit(f"Missing {new_best}")

    if NER_MODEL_BEST.exists():
        shutil.rmtree(NER_MODEL_BEST)
    shutil.copytree(new_best, NER_MODEL_BEST)
    shutil.rmtree(NER_TMP, ignore_errors=True)
    tmp_cfg.unlink(missing_ok=True)
    print(f"NER model deployed to {NER_MODEL_BEST}")


if __name__ == "__main__":
    main()
