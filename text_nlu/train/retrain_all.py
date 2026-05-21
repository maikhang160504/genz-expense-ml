"""
Huấn luyện lại toàn bộ mô hình TF-IDF (.joblib) + chuẩn bị & train NER spaCy (thư mục ``text_nlu``).

Thứ tự:
1) train_intent_model
2) train_record_type_model
3) train_category_model
4) train_action_type_model
5) ner_prepare (jsonl -> .spacy)
7) spacy train -> ghi đè models/ner_model/model-best

Biến môi trường:
- NER_MAX_STEPS: mặc định 6000 (giảm nếu cần train nhanh).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent


def _run(script: str) -> None:
    path = TRAIN_DIR / script
    print(f"\n>>> python {path.name}")
    subprocess.run([sys.executable, str(path)], cwd=str(TRAIN_DIR), check=True)


def main() -> None:
    os.chdir(TRAIN_DIR)
    _run("train_intent_model.py")
    _run("train_record_type_model.py")
    _run("train_category_model.py")
    _run("train_action_type_model.py")
    _run("ner_prepare.py")
    _run("train_ner_only.py")
    print("All training steps finished.")


if __name__ == "__main__":
    main()
