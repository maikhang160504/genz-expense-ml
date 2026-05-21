"""
Demo local: FastAPI nhận ảnh hóa đơn → OCR → danh mục + số tiền (NLU).

Chạy (từ thư mục Train):
  pip install -r OCR/requirements.txt
  set RECEIPT_OCR_WEIGHTS=OCR/models/vietocr_receipt.pth
  uvicorn OCR.demo.server:app --reload --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
OCR_ROOT = ROOT / "OCR"
sys.path.insert(0, str(OCR_ROOT / "src"))

from receipt_ocr.pipeline import ReceiptOCRPipeline  # noqa: E402

app = FastAPI(title="Vietnamese Receipt OCR Demo", version="2.0.0")

DEFAULT_WEIGHTS = OCR_ROOT / "models" / "vietocr_receipt.pth"
WEIGHTS_PATH = Path(os.environ.get("RECEIPT_OCR_WEIGHTS", str(DEFAULT_WEIGHTS)))

_pipeline: ReceiptOCRPipeline | None = None


def get_pipeline() -> ReceiptOCRPipeline:
    global _pipeline
    if _pipeline is None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"Không thấy weights: {WEIGHTS_PATH}. "
                "Tải artifact từ Kaggle hoặc đặt biến RECEIPT_OCR_WEIGHTS."
            )
        _pipeline = ReceiptOCRPipeline(WEIGHTS_PATH).load()
    return _pipeline


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8"/>
  <title>Demo OCR Hóa đơn VN</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
    pre { background: #f4f4f5; padding: 1rem; overflow: auto; border-radius: 8px; }
    .hint { color: #555; font-size: 0.95rem; }
  </style>
</head>
<body>
  <h1>Nhận dạng hóa đơn</h1>
  <p class="hint">Trả về <strong>category</strong> (NLU Record: Food, Shopping, …) + <strong>amount</strong> (VND).</p>
  <form id="f">
    <input type="file" name="file" accept="image/*" required />
    <button type="submit">Phân tích</button>
  </form>
  <pre id="out">Chọn ảnh và bấm Phân tích…</pre>
  <script>
    document.getElementById('f').onsubmit = async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target);
      const out = document.getElementById('out');
      out.textContent = 'Đang xử lý…';
      const res = await fetch('/ocr', { method: 'POST', body: fd });
      out.textContent = JSON.stringify(await res.json(), null, 2);
    };
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok", "weights": str(WEIGHTS_PATH), "exists": WEIGHTS_PATH.is_file()}


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = get_pipeline().process_image(tmp_path)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
