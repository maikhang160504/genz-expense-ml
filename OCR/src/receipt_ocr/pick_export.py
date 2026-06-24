"""Export admin-verified labels to PICK TSV for Kaggle retrain."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def box_to_pick_line(text: str, bbox: list[int | float], entity: str) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    clean = (text or "").replace("\t", " ").strip()
    ent = entity if entity else "OTHER"
    return f"{x1}\t{y1}\t{x2}\t{y2}\t{clean}\t{ent}"


def sample_to_pick_tsv(sample_id: str, boxes: list[dict[str, Any]]) -> str:
    lines = ["x1\ty1\tx2\ty2\ttext\tentity"]
    for box in boxes:
        bbox = box.get("bbox") or [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
        if not bbox or len(bbox) < 4:
            continue
        text = str(box.get("text", ""))
        entity = str(box.get("entity", "OTHER"))
        lines.append(box_to_pick_line(text, bbox, entity))
    return "\n".join(lines) + "\n"


def export_verified_samples(
    samples: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    samples: [{id, boxes: [{text,x1,y1,x2,y2,entity}, ...]}]
    Writes boxes_and_transcripts/{id}.tsv and manifest.json
    """
    out = Path(output_dir)
    tsv_dir = out / "boxes_and_transcripts"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"samples": [], "count": 0}

    for sample in samples:
        sid = str(sample.get("id") or sample.get("sample_id", "")).strip()
        if not sid:
            continue
        boxes = sample.get("admin_labels", sample.get("boxes", []))
        if not boxes:
            continue
        tsv_path = tsv_dir / f"{sid}.tsv"
        tsv_path.write_text(sample_to_pick_tsv(sid, boxes), encoding="utf-8")
        manifest["samples"].append({"id": sid, "tsv": str(tsv_path.name)})
        manifest["count"] += 1

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(out), "manifest": manifest, "manifest_path": str(manifest_path)}

