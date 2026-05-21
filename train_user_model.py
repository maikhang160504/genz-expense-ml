#!/usr/bin/env python3
"""
train_user_model.py — TASK-08: Per-user model training from corrections.

Reads user corrections from the backend PostgreSQL database, builds a
per-user category mapping overlay, and optionally fine-tunes the TF-IDF
intent/category models for a specific user.

Usage:
    python train_user_model.py --user-id <UUID> [--db-url <DATABASE_URL>]
    python train_user_model.py --all [--db-url <DATABASE_URL>]

The generated per-user models are saved to:
    text_nlu/models/user_<user_id>/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib
    import psycopg2
except ImportError:
    print("Missing dependencies. Install with: pip install joblib psycopg2-binary")
    sys.exit(1)


DEFAULT_DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://postgres:postgres@localhost:5432/expense_ai",
)

MODELS_DIR = PROJECT_ROOT / "text_nlu" / "models"


def fetch_corrections(db_url: str, user_id: str | None = None) -> list[dict]:
    """Fetch user_corrections rows from DB."""
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT user_id, text, intent, category_code, record_type,
                           action_type, predicted, source, created_at
                    FROM user_corrections
                    WHERE user_id = %s
                    ORDER BY created_at
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT user_id, text, intent, category_code, record_type,
                           action_type, predicted, source, created_at
                    FROM user_corrections
                    ORDER BY user_id, created_at
                    """
                )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def build_category_mappings(corrections: list[dict]) -> dict[str, str]:
    """
    Build a text → category_code mapping from corrections.
    When a user corrects the category, we learn that association.
    """
    mappings: dict[str, str] = {}
    for c in corrections:
        if c.get("text") and c.get("category_code"):
            key = c["text"].strip().lower()
            mappings[key] = c["category_code"]
    return mappings


def build_intent_overrides(corrections: list[dict]) -> dict[str, str]:
    """
    Build a text → intent mapping from corrections where user
    changed the intent (e.g., AI said Chitchat but user meant Record).
    """
    overrides: dict[str, str] = {}
    for c in corrections:
        if c.get("text") and c.get("intent"):
            predicted = c.get("predicted")
            if predicted:
                try:
                    pred_data = json.loads(predicted) if isinstance(predicted, str) else predicted
                    if pred_data.get("intent") != c["intent"]:
                        overrides[c["text"].strip().lower()] = c["intent"]
                except (json.JSONDecodeError, TypeError):
                    pass
    return overrides


def save_user_model(user_id: str, category_mappings: dict, intent_overrides: dict):
    """Save per-user model artifacts."""
    user_dir = MODELS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    # Save category mappings
    mappings_path = user_dir / "category_mappings.json"
    with open(mappings_path, "w", encoding="utf-8") as f:
        json.dump(category_mappings, f, ensure_ascii=False, indent=2)

    # Save intent overrides
    overrides_path = user_dir / "intent_overrides.json"
    with open(overrides_path, "w", encoding="utf-8") as f:
        json.dump(intent_overrides, f, ensure_ascii=False, indent=2)

    # Save metadata
    meta = {
        "user_id": user_id,
        "num_category_mappings": len(category_mappings),
        "num_intent_overrides": len(intent_overrides),
        "version": "1.0.0",
    }
    meta_path = user_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved model for user {user_id}:")
    print(f"     - {len(category_mappings)} category mappings")
    print(f"     - {len(intent_overrides)} intent overrides")
    print(f"     - Path: {user_dir}")
    return user_dir


def train_for_user(db_url: str, user_id: str):
    """Train a per-user model from their corrections."""
    corrections = fetch_corrections(db_url, user_id)
    if not corrections:
        print(f"[SKIP] No corrections found for user {user_id}")
        return

    print(f"[INFO] Found {len(corrections)} corrections for user {user_id}")

    category_mappings = build_category_mappings(corrections)
    intent_overrides = build_intent_overrides(corrections)

    if not category_mappings and not intent_overrides:
        print(f"[SKIP] No actionable corrections for user {user_id}")
        return

    save_user_model(user_id, category_mappings, intent_overrides)


def train_all(db_url: str):
    """Train per-user models for all users with corrections."""
    corrections = fetch_corrections(db_url)
    if not corrections:
        print("[SKIP] No corrections found in database")
        return

    # Group by user_id
    users: dict[str, list[dict]] = {}
    for c in corrections:
        uid = str(c["user_id"])
        users.setdefault(uid, []).append(c)

    print(f"[INFO] Found {len(corrections)} total corrections for {len(users)} users")

    for uid, user_corrections in users.items():
        category_mappings = build_category_mappings(user_corrections)
        intent_overrides = build_intent_overrides(user_corrections)
        if category_mappings or intent_overrides:
            save_user_model(uid, category_mappings, intent_overrides)


def main():
    parser = argparse.ArgumentParser(description="Train per-user NLU models from corrections")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user-id", type=str, help="Train model for specific user UUID")
    group.add_argument("--all", action="store_true", help="Train models for all users with corrections")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="PostgreSQL connection URL")
    args = parser.parse_args()

    if args.all:
        train_all(args.db_url)
    else:
        train_for_user(args.db_url, args.user_id)


if __name__ == "__main__":
    main()
