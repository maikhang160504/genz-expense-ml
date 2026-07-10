# LayoutLMv3 Training & Inference Guide

This guide explains how to preprocess data, manage checkpoints, and run the LayoutLMv3 model for expense-receipt OCR and Key Information Extraction (KIE) using the codebase under `bill_ocr/layoutlmv3`.

---

## 1️⃣ Prerequisites

Install the project dependencies (including the LayoutLMv3 stack):

```bash
# From the repository root (or expense-ocr-nlu directory)
pip install -r requirements.txt
# Additional packages required for LayoutLMv3 (if not already in requirements.txt)
pip install transformers datasets seqeval accelerate
```

---

## 2️⃣ Directory Layout

```
expense-ocr-nlu/
└─ bill_ocr/
   └─ layoutlmv3/
      ├─ __init__.py
      ├─ configs/
      │   └─ layoutlmv3.yaml          # default hyper‑parameters & paths
      ├─ src/
      │   ├─ config.py                # loads yaml → dataclass
      │   ├─ data.py                  # dataset handling
      │   └─ model.py                 # processor, model, label maps
      ├─ scripts/
      │   ├─ filter_layoutlmv3_dataset.py  # [NEW] Preprocesses and cleans raw dataset
      │   └─ convert_csv_to_jsonl.py  # Converts CSV → JSONL with dynamic OCR for test set
      └─ train_eval.py                # Training, validation & test pipeline script
```

---

## 3️⃣ Dataset Preprocessing & Cleaning

Before training, you must clean the raw MC_OCR dataset to fix misspelled labels and drop rows with mismatched polygon/transcription lists.

### How it works:
- Automatically fixes label typos: `TIMESTAMPS -> TIMESTAMP` and `TOTAL_TOTAL_COST -> TOTAL_COST`.
- Drops mismatched rows (e.g., rows with 0 polygons or list length differences) which would pollute the training sequence.
- Filters out receipts with too few unique fields (< 3) or too many total cost entries (> 4).
- Saves only the validated matching images to the output directory.

### Run locally (One-line PowerShell/Bash):
```bash
# Make sure you are in d:\Luan-Van\Project\expense-ocr-nlu
python -m bill_ocr.layoutlmv3.scripts.filter_layoutlmv3_dataset --csv data/mcocr_train_df.csv --img_dir data/mc_ocr_train --out_csv data/mcocr_train_df_layoutlmv3_cleaned.csv --out_img_dir data/mc_ocr_train_layoutlmv3_cleaned
```

---

## 4️⃣ Uploading to Modal Volume

To keep the LayoutLMv3 training data separate from standard PICK data, we upload the cleaned dataset to dedicated paths on Modal.

Run the following commands from `d:\Luan-Van\Project\expense-ocr-nlu` to upload:

```bash
# 1. Upload the cleaned CSV file
modal volume put -f expense-ocr-nlu-storage data/mcocr_train_df_layoutlmv3_cleaned.csv /layoutlmv3_train_df.csv

# 2. Upload the cleaned images folder
modal volume put -f expense-ocr-nlu-storage data/mc_ocr_train_layoutlmv3_cleaned /layoutlmv3_train_imgs
```

---

## 5️⃣ Configuration (`layoutlmv3.yaml`)

The training parameters are defined in `configs/layoutlmv3.yaml`. Notice the training paths now point to the separated cleaned directories:

```yaml
epochs: 5
learning_rate: 5e-5
seed: 42
batch_size: 4
early_stop_patience: 3
max_seq_length: 512
num_workers: 2

train:
  images_dir: /storage/layoutlmv3_train_imgs
  csv_path: /storage/layoutlmv3_train_df.csv
  output_dir: /workspace/bill_ocr/layoutlmv3/data/train

val:
  images_dir: /storage/layoutlmv3_train_imgs
  csv_path: /storage/layoutlmv3_train_df.csv
  output_dir: /workspace/bill_ocr/layoutlmv3/data/val

test:
  images_dir: /storage/mc_ocr_test/test_images
  csv_path: /storage/mcocr_test_df.csv
  output_dir: /workspace/bill_ocr/layoutlmv3/data/test
```

---

## 6️⃣ Running LayoutLMv3 on Modal

### 1. Training (and Resuming Checkpoints)
If a training session timeouts or gets disconnected, you can resume it using the `--resume-from-checkpoint` parameter:

```bash
# Train from scratch (default 15 epochs)
modal run modal_app.py::train_layoutlmv3_model

# Resume training from a saved checkpoint folder
modal run modal_app.py::train_layoutlmv3_model --resume-from-checkpoint="/storage/saved_checkpoint/checkpoint-500"
```

> [!NOTE]
> When resuming training, the script reloads the exact same `train_split.jsonl` and `val_split.jsonl` generated in the first run to prevent data leakage and ensure train/validation consistency.

### 2. Evaluation
To compute the validation F1 score (using seqeval) on the 10% validation split:
```bash
modal run --detach modal_app.py::evaluate_layoutlmv3_model
```
modal run modal_app.py::visualize_layoutlmv3_test_predictions

### 3. Inference / Testing on Unlabeled Data
When running test inference, `test_df.csv` is just a list of images without annotation labels. The pipeline automatically runs the project's PaddleOCR + VietOCR pipeline on-the-fly to extract bounding boxes and text, normalize coordinates to 0-1000, and save results.

To generate the predictions CSV `result.csv`:
```bash
modal run modal_app.py::test_layoutlmv3_model
```

---

## 7️⃣ Expected Outputs

All checkpoints and evaluation logs survive container restarts since they are synced back to the persistent `/storage` volume:
- **Best model weights**: `/storage/layoutlmv3/model_best.pth` (Also kept in container workspace at `/workspace/bill_ocr/models/layoutlmv3/model_best.pth`)
- **Evaluation F1 Metrics**: `/storage/evaluation_metrics_layoutlmv3.txt`
- **Private test predictions**: `/storage/result.csv`
