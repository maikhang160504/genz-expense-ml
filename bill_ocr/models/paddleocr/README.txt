PaddleOCR detection weights are NOT stored in this repo.

On first run, PaddleOCR 3.x downloads PP-OCRv5_mobile_det to:
  Windows: C:\Users\<user>\.paddlex\official_models\PP-OCRv5_mobile_det
  Linux/Kaggle: ~/.paddlex/official_models/PP-OCRv5_mobile_det

Configured in: OCR/src/receipt_ocr/paddle_compat.py (detection-only, lang=vi).

No fine-tune — pretrained only (MC_OCR step 1).
