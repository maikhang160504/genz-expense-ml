"""
FastAPI application entrypoint for the unified expense-ocr-nlu service.
Wires path dependencies and instantiates the main application.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve root directory of expense-ocr-nlu
ROOT_DIR = Path(__file__).resolve().parents[2]

# Insert paths to sys.path so NLU, OCR and app packages import correctly
sys.path.insert(0, str(ROOT_DIR / "src" / "api"))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "text_nlu"))

# Import main create_app factory from the merged app folder
from app.main import create_app

app = create_app()
