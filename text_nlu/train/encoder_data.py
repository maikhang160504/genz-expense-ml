"""Shared dataset loaders for PhoBERT encoder training (aligned with current CSV schema)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"

RECORD_COLUMNS = ["text", "label", "type", "is_money"]
ACTION_COLUMNS = ["text", "intent", "action_type"]


def _read_csv(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols)


def subsample_by_label(
    df: pd.DataFrame,
    label_col: str,
    max_samples: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    n_labels = max(1, df[label_col].nunique())
    per = max(1, max_samples // n_labels)
    parts = [
        g.sample(n=min(len(g), per), random_state=seed)
        for _, g in df.groupby(label_col, group_keys=False)
    ]
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_samples:
        out = out.sample(n=max_samples, random_state=seed)
    return out.reset_index(drop=True)


def load_intent_rows(*, max_samples: int | None = None) -> pd.DataFrame:
    """Record + Action + Chitchat → intent column (Record / Action / Chitchat)."""
    rec = _read_csv("intent_record.csv", usecols=["text"]).copy()
    rec["intent"] = "Record"
    act = _read_csv("intent_action.csv", usecols=["text", "intent"])
    chat = _read_csv("intent_chitchat.csv", usecols=["text", "intent"])
    df = pd.concat([rec, act, chat], ignore_index=True)
    df.dropna(subset=["text", "intent"], inplace=True)

    n_record = int((df["intent"] == "Record").sum())
    n_action = int((df["intent"] == "Action").sum())
    n_chat = int((df["intent"] == "Chitchat").sum())

    action_target = max(n_action, int(n_record * 0.18), 8000)
    action_df = df[df["intent"] == "Action"]
    if len(action_df) < action_target:
        extra = action_df.sample(n=action_target - len(action_df), replace=True, random_state=42)
        df = pd.concat([df, extra], ignore_index=True)

    chat_target = min(len(df[df["intent"] == "Action"]), max(n_chat * 2, 2000))
    chat_df = df[df["intent"] == "Chitchat"]
    if len(chat_df) < chat_target:
        extra = chat_df.sample(n=chat_target - len(chat_df), replace=True, random_state=42)
        df = pd.concat([df, extra], ignore_index=True)

    cap = max_samples if max_samples is not None else int(os.environ.get("INTENT_ENCODER_MAX_SAMPLES", "0"))
    if cap > 0:
        df = subsample_by_label(df, "intent", cap)
    return df


def load_action_type_rows(*, max_samples: int | None = None) -> pd.DataFrame:
    """Action utterances → fine-grained action_type (SET_LIMIT, ADD_GOAL, …)."""
    df = _read_csv("intent_action.csv", usecols=["text", "action_type"])
    df = df.dropna(subset=["text", "action_type"])
    df["action_type"] = df["action_type"].astype(str).str.strip().str.upper()
    cap = max_samples if max_samples is not None else int(os.environ.get("ACTION_TYPE_ENCODER_MAX_SAMPLES", "0"))
    if cap > 0:
        df = subsample_by_label(df, "action_type", cap)
    return df


def load_record_type_rows(*, max_samples: int | None = None) -> pd.DataFrame:
    df = _read_csv("intent_record.csv", usecols=["text", "type"])
    df = df.dropna(subset=["text", "type"])
    df["type"] = df["type"].astype(str).str.lower()
    cap = max_samples if max_samples is not None else int(os.environ.get("RECORD_TYPE_ENCODER_MAX_SAMPLES", "0"))
    if cap > 0:
        df = subsample_by_label(df, "type", cap)
    return df


def load_category_rows(*, max_samples: int | None = None) -> pd.DataFrame:
    df = _read_csv("intent_record.csv", usecols=["text", "label"])
    df = df.dropna(subset=["text", "label"])
    cap = max_samples if max_samples is not None else int(os.environ.get("CATEGORY_ENCODER_MAX_SAMPLES", "0"))
    if cap > 0:
        df = subsample_by_label(df, "label", cap)
    return df


def xy_from_df(df: pd.DataFrame, label_col: str) -> tuple[list[str], np.ndarray]:
    texts = df["text"].astype(str).tolist()
    y = np.asarray(df[label_col].astype(str).tolist(), dtype=str)
    return texts, y
