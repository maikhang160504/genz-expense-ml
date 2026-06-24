PICK KIE model (MC_OCR style)

Deploy from Kaggle:
  model_best.pth  →  this folder/
  config.json     →  optional

Runtime inference uses MC_OCR PICK code:
  ../../MC_OCR/mc_ocr/key_info_extraction/PICK
  or env MC_OCR_PICK_DIR=/path/to/PICK

Optional pip deps for PICK inference:
  torchtext==0.6.0
  overrides

Until model_best.pth exists, ai-service uses heuristic KIE fallback.
