#!/usr/bin/env python3
"""Visualize bill-demo images with two OCR/KIE pipelines side by side.

Pipeline A — KIE-only (OCR → PICK/heuristic, no fusion):
  Same word boxes as training input; KIE labels only.

Pipeline B — Production MC-OCR (OCR → parallel KIE + prep → fusion):

Usage (from expense-ocr-nlu, with venv active):
  python OCR/tools/visualize_bill_demo.py
  python OCR/tools/visualize_bill_demo.py --open
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

OCR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr.hybrid_pipeline import HybridReceiptOCRPipeline  # noqa: E402
from receipt_ocr.model_paths import PICK_KIE_MODEL_PATH, resolve_vietocr_weights_path  # noqa: E402
from receipt_ocr.pick_kie import get_kie_engine, pick_kie_weights_status  # noqa: E402
from receipt_ocr.receipt_fusion import run_hybrid_pipeline, run_kie_branch, run_ocr_stage  # noqa: E402

BILL_DEMO_DIR = OCR_ROOT / "tests" / "bill-demo"
OUTPUT_DIR = BILL_DEMO_DIR / "output"

ENTITY_COLORS_BGR = {
    "SELLER": (246, 130, 59),
    "ADDRESS": (184, 163, 148),
    "TIMESTAMP": (22, 115, 249),
    "TOTAL_COST": (129, 185, 16),
    "OTHER": (8, 179, 234),
}


def list_demo_images() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in BILL_DEMO_DIR.iterdir() if p.suffix.lower() in exts and p.is_file())


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: list[dict],
    *,
    title: str,
    meta_lines: list[str] | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    for box in boxes:
        x1 = int(box.get("x1", box.get("bbox", [0, 0, 0, 0])[0]))
        y1 = int(box.get("y1", box.get("bbox", [0, 0, 0, 0])[1]))
        x2 = int(box.get("x2", box.get("bbox", [0, 0, 0, 0])[2]))
        y2 = int(box.get("y2", box.get("bbox", [0, 0, 0, 0])[3]))
        ent = str(box.get("entity", "OTHER"))
        color = ENTITY_COLORS_BGR.get(ent, ENTITY_COLORS_BGR["OTHER"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{ent}"
        text = str(box.get("text", "")).strip()
        if text:
            label = f"{ent}: {text[:28]}{'…' if len(text) > 28 else ''}"
        ty = max(y1 - 6, 14)
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    bar_h = 56 + (18 * len(meta_lines or []))
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (20, 27, 43)
    cv2.putText(bar, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 252), 2, cv2.LINE_AA)
    for i, line in enumerate(meta_lines or []):
        cv2.putText(
            bar,
            line,
            (12, 42 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([bar, canvas])


def run_kie_only(pipeline: HybridReceiptOCRPipeline, image_rgb: np.ndarray) -> dict:
    """OCR word boxes → PICK/heuristic KIE only (no fusion)."""
    ocr = run_ocr_stage(pipeline, image_rgb)
    kie_engine = pipeline._get_kie()
    kie = run_kie_branch(kie_engine, ocr.df_boxes, image_rgb)
    return {
        "boxes": kie.labeled_boxes,
        "kie_fields": kie.kie_fields,
        "kie_backend": kie.kie_backend,
        "box_count": len(kie.labeled_boxes),
        "ocr_boxes": len(ocr.df_boxes),
    }


def run_production_hybrid(pipeline: HybridReceiptOCRPipeline, image_rgb: np.ndarray) -> dict:
    """Full MC-OCR pipeline with fusion."""
    fused = run_hybrid_pipeline(pipeline, pipeline._get_kie(), image_rgb, parallel=True)
    summary = fused.summary
    return {
        "boxes": summary.get("boxes", fused.labeled_boxes),
        "kie_fields": summary.get("kie_fields", fused.kie_fields),
        "kie_backend": summary.get("kie_backend", fused.kie_backend),
        "category": summary.get("category"),
        "amount": summary.get("amount"),
        "items_count": summary.get("items_count", 0),
        "warnings": summary.get("warnings") or [],
    }


def build_html(report: dict, output_dir: Path) -> Path:
    rows = []
    for item in report["images"]:
        name = item["name"]
        a = item["kie_only"]["file"]
        b = item["hybrid"]["file"]
        rows.append(f"""
        <section class="card">
          <h2>{name}</h2>
          <div class="grid">
            <figure>
              <figcaption>Pipeline A — KIE only (OCR → PICK/heuristic)</figcaption>
              <img src="{a}" alt="{name} kie" />
              <pre>{json.dumps(item["kie_only"]["summary"], ensure_ascii=False, indent=2)}</pre>
            </figure>
            <figure>
              <figcaption>Pipeline B — Production (KIE + fusion)</figcaption>
              <img src="{b}" alt="{name} hybrid" />
              <pre>{json.dumps(item["hybrid"]["summary"], ensure_ascii=False, indent=2)}</pre>
            </figure>
          </div>
        </section>""")

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>Bill demo — pipeline comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0b0f19; color: #e2e8f0; margin: 0; padding: 24px; }}
    h1 {{ margin-top: 0; }}
    .meta {{ color: #94a3b8; margin-bottom: 24px; }}
    .legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
    .legend span {{ padding: 4px 10px; border-radius: 999px; font-size: 12px; }}
    .card {{ background: #141b2c; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-weight: 600; margin-bottom: 8px; color: #a7f3d0; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #334155; }}
    pre {{ background: #070a13; padding: 12px; border-radius: 8px; font-size: 12px; overflow: auto; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Bill demo — 2 pipeline comparison</h1>
  <p class="meta">Generated {report["generated_at"]} · PICK weights: {report["kie_status"]}</p>
  <div class="legend">
    <span style="background:#3b82f633;color:#93c5fd">SELLER</span>
    <span style="background:#94a3b833;color:#cbd5e1">ADDRESS</span>
    <span style="background:#f9731633;color:#fdba74">TIMESTAMP</span>
    <span style="background:#10b98133;color:#6ee7b7">TOTAL_COST</span>
    <span style="background:#eab30833;color:#fde047">OTHER</span>
  </div>
  {''.join(rows)}
</body>
</html>"""
    out = output_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize bill-demo with two OCR pipelines")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    args = parser.parse_args()

    images = list_demo_images()
    if not images:
        print(f"No images in {BILL_DEMO_DIR}")
        return 1

    weights = resolve_vietocr_weights_path()
    if not weights.is_file():
        print(f"VietOCR weights missing: {weights}")
        return 1

    kie_status = pick_kie_weights_status(PICK_KIE_MODEL_PATH)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    print(f"Loading MC-OCR pipeline (device={device}) …")
    pipeline = HybridReceiptOCRPipeline(
        vietocr_weights=weights,
        device=device,
        pick_kie_model=PICK_KIE_MODEL_PATH,
    ).load()
    get_kie_engine(model_path=PICK_KIE_MODEL_PATH, device=device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kie_status": kie_status,
        "device": device,
        "images": [],
    }

    for img_path in images:
        print(f"Processing {img_path.name} …")
        image_rgb = pipeline._read_rgb(img_path)

        kie = run_kie_only(pipeline, image_rgb)
        hy = run_production_hybrid(pipeline, image_rgb)

        kie_meta = [
            f"KIE backend: {kie['kie_backend']}",
            f"OCR boxes: {kie['ocr_boxes']} · labeled: {kie['box_count']}",
            f"SELLER: {(kie['kie_fields'] or {}).get('SELLER') or '—'}",
            f"TOTAL: {(kie['kie_fields'] or {}).get('TOTAL_COST_VALUE') or '—'}",
        ]
        hy_meta = [
            f"KIE backend: {hy['kie_backend']}",
            f"Category: {hy.get('category') or '—'} · Amount: {hy.get('amount') or '—'}",
            f"Items: {hy.get('items_count', 0)} · boxes: {len(hy.get('boxes') or [])}",
        ]

        stem = img_path.stem
        kie_file = f"{stem}_pipeline_a_kie.jpg"
        hy_file = f"{stem}_pipeline_b_hybrid.jpg"

        kie_img = draw_boxes(
            image_rgb,
            kie["boxes"],
            title="Pipeline A — KIE only (OCR → PICK/heuristic)",
            meta_lines=kie_meta,
        )
        hy_img = draw_boxes(
            image_rgb,
            hy["boxes"],
            title="Pipeline B — Production (OCR → KIE + fusion)",
            meta_lines=hy_meta,
        )

        cv2.imwrite(str(OUTPUT_DIR / kie_file), kie_img)
        cv2.imwrite(str(OUTPUT_DIR / hy_file), hy_img)

        report["images"].append({
            "name": img_path.name,
            "kie_only": {
                "file": kie_file,
                "summary": {
                    "kie_backend": kie["kie_backend"],
                    "kie_fields": kie["kie_fields"],
                    "box_count": kie["box_count"],
                },
            },
            "hybrid": {
                "file": hy_file,
                "summary": {
                    "kie_backend": hy["kie_backend"],
                    "category": hy.get("category"),
                    "amount": hy.get("amount"),
                    "kie_fields": hy.get("kie_fields"),
                    "items_count": hy.get("items_count"),
                    "warnings": hy.get("warnings"),
                },
            },
        })

    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = build_html(report, OUTPUT_DIR)
    print(f"\nDone — {len(images)} image(s)")
    print(f"  Annotated JPGs: {OUTPUT_DIR}")
    print(f"  HTML report:    {html_path}")
    if args.open:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
