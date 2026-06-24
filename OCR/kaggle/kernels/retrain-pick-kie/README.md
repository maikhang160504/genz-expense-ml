# retrain-pick-kie

Fine-tune **PICK KIE** trên MC-OCR base + nhãn incremental WebAdmin.

**Code:** `vendor_pick.zip` (`OCR/vendor/pick`) — không clone MC_OCR.

## Notebook cells

1. Bootstrap common · 2. Materialize PICK · 3. Dataset paths · 4. Build TSV · 5. Merge WebAdmin · 6. Train · 7. Export

## Flow

1. Build PICK data từ `mcocr_train_df.csv` (giống train kernel)
2. Merge `webadmin-verified-receipts/incremental/`:
   - `boxes_and_transcripts/*.tsv`
   - `images/*`
3. Train PICK (default 25 epochs — set env `PICK_EPOCHS`)
4. Export `pick_kie_artifacts/model_best.pth`

## Kaggle setup

Add Data:

- `domixi1989/vietnamese-receipts-mc-ocr-2021`
- `mainhatkhangb2205881/webadmin-verified-receipts` (sau `kaggle datasets version`)

```bash
kaggle datasets version -p expense-ocr-nlu/OCR/verified_ocr_labels/kaggle_upload
kaggle kernels push -p expense-ocr-nlu/OCR/kaggle/kernels/retrain-pick-kie
```

## Deploy

`POST /bill-retrain/kaggle/deploy` with `job_type=pick_retrain`
