"""Golden test fixtures for end-to-end bill OCR eval (never used in training)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "tests" / "golden" / "fixtures.jsonl"


def load_golden_fixtures(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or FIXTURES_PATH
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def eval_summary_against_golden(
    predicted: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, bool]:
    amount_ok = predicted.get("amount") == expected.get("amount")
    category_ok = predicted.get("category") == expected.get("category")
    seller_ok = True
    if expected.get("seller_contains"):
        seller = str((predicted.get("kie_fields") or {}).get("SELLER") or predicted.get("seller") or "")
        seller_ok = expected["seller_contains"].lower() in seller.lower()
    return {"amount_exact": amount_ok, "category_top1": category_ok, "seller_match": seller_ok}


def run_golden_eval(results: list[dict[str, Any]]) -> dict[str, Any]:
    """results: [{fixture_id, predicted, expected}]"""
    if not results:
        return {"n": 0, "amount_acc": 0.0, "category_acc": 0.0, "pass": False}
    amount_hits = sum(1 for r in results if r["metrics"]["amount_exact"])
    cat_hits = sum(1 for r in results if r["metrics"]["category_top1"])
    n = len(results)
    amount_acc = amount_hits / n
    category_acc = cat_hits / n
    return {
        "n": n,
        "amount_acc": round(amount_acc, 4),
        "category_acc": round(category_acc, 4),
        "pass": amount_acc >= 0.85 and category_acc >= 0.80,
        "details": results,
    }
