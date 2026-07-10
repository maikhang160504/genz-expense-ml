from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def export_verified_samples(samples: list[dict[str, Any]], incremental_dir: Path) -> dict[str, Any]:
    """
    Exports verified bounding boxes and transcriptions for PICK KIE.
    Each sample is saved as a TSV file in incremental_dir/boxes_and_transcripts/{id}.tsv
    Format: index \t x0 \t y0 \t x1 \t y1 \t x2 \t y2 \t x3 \t y3 \t text \t label
    """
    tsv_dir = incremental_dir / "boxes_and_transcripts"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for sample in samples:
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            continue
            
        # Retrieve the annotations
        boxes = sample.get("adminLabels") or sample.get("admin_labels")
        if not boxes:
            auto_labels = sample.get("autoLabels") or sample.get("auto_labels") or {}
            boxes = auto_labels.get("boxes", []) if isinstance(auto_labels, dict) else []
            
        tsv_path = tsv_dir / f"{sample_id}.tsv"
        try:
            with open(tsv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                for idx, box in enumerate(boxes):
                    # Extract coordinates
                    x1 = int(box.get("x1", 0))
                    y1 = int(box.get("y1", 0))
                    x2 = int(box.get("x2", 0))
                    y2 = int(box.get("y2", 0))
                    
                    # Safe transcription clean up
                    text = str(box.get("text", "")).strip().replace("\t", " ").replace("\n", " ")
                    
                    # Entity label normalization
                    label = str(box.get("entity") or box.get("label") or "OTHER").strip().upper()
                    if label == "O":
                        label = "OTHER"
                    
                    # 4 points layout (clockwise): (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                    writer.writerow([
                        idx,
                        x1, y1,
                        x2, y1,
                        x2, y2,
                        x1, y2,
                        text,
                        label
                    ])
            count += 1
        except Exception as e:
            logger.error(f"Failed to export TSV for sample {sample_id}: {e}")
            
    return {
        "manifest": {
            "count": count
        }
    }
