"""Load action_slot_columns without conflicting with HuggingFace ``datasets`` on Kaggle."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SCHEMA_PATH = Path(__file__).resolve().parent / "datasets" / "action_slot_columns.py"


def import_slot_schema() -> ModuleType:
    if not _SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Missing slot schema: {_SCHEMA_PATH}")
    spec = importlib.util.spec_from_file_location("action_slot_columns", _SCHEMA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load slot schema from {_SCHEMA_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
