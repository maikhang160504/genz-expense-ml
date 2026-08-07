# LayoutLMv3 training, evaluation and inference pipeline
"""Complete end‑to‑end script for LayoutLMv3 on the invoice KIE dataset.

Features
--------
1️⃣ **Data preparation** – copies images & CSV from Modal volume, converts CSV → JSONL
   (required format for the LayoutLMv3 processor).
2️⃣ **Training** – uses 🤗 `Trainer` (with `accelerate` support) to fine‑tune the model.
3️⃣ **Evaluation** – loads the best checkpoint and computes F1 (seqeval).
4️⃣ **Test / inference** – runs the model on a test split and writes predictions
   to ``result.csv`` (columns: ``image_path``, ``predicted_entities``).

The script can be invoked from the container (or via Modal) as:

```bash
python -m bill_ocr.mc_ocr.layoutlmv3.train_eval train   # train the model
python -m bill_ocr.mc_ocr.layoutlmv3.train_eval eval    # evaluate on validation split
python -m bill_ocr.mc_ocr.layoutlmv3.train_eval test    # run inference & export CSV
```
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (container side)
# ---------------------------------------------------------------------------
if os.path.exists("/workspace"):
    DATA_ROOT = Path("/workspace/data")
else:
    DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
VOLUME_TRAIN_IMG = Path("/storage/layoutlmv3_train_imgs")
VOLUME_TRAIN_CSV = Path("/storage/layoutlmv3_train_df.csv")
# Optional test data – can be omitted; if present, Modal volume should contain the files.
VOLUME_TEST_IMG = Path("/storage/mc_ocr_test")
VOLUME_TEST_CSV = Path("/storage/mcocr_test_df.csv")

CHECKPOINT_PATH = Path("/workspace/bill_ocr/models/layoutlmv3/model_best.pth")
VOLUME_CHECKPOINT = Path("/storage/layoutlmv3/candidate_model.pth")
RESULTS_PATH = Path("/storage/evaluation_metrics_layoutlmv3.txt")
TEST_RESULT_CSV = Path("/storage/result.csv")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def copy_dataset(train: bool = True):
    """Copy images & CSV from the Modal volume to the workspace.
    If ``train`` is ``True`` copies the training split, otherwise copies the test split.
    """
    split = "train" if train else "test"
    target_dir = DATA_ROOT / split
    target_dir.mkdir(parents=True, exist_ok=True)
    # Images
    src_img = VOLUME_TRAIN_IMG if train else VOLUME_TEST_IMG
    shutil.copytree(src_img, target_dir / "imgs", dirs_exist_ok=True)
    # CSV
    src_csv = VOLUME_TRAIN_CSV if train else VOLUME_TEST_CSV
    shutil.copy2(src_csv, target_dir / f"{split}_df.csv")
    print(f"✅ {split.capitalize()} dataset copied to {target_dir}")

def convert_csv_to_jsonl(csv_path: Path, jsonl_dir: Path):
    """Run the conversion script located in ``layoutlmv3/scripts``.
    The script expects columns ``image_path``, ``bbox`` (semicolon separated
    ``x0,y0,x1,y1``) and ``label`` (semicolon separated IOB tags). It creates a
    JSONL file where each line contains ``image``, ``words``, ``boxes`` and
    ``labels``.
    """
    script_path = Path(__file__).parent / "scripts" / "convert_csv_to_jsonl.py"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python",
        str(script_path),
        "--csv",
        str(csv_path),
        "--out",
        str(jsonl_dir),
    ])
    print(f"✅ CSV -> JSONL conversion completed: {jsonl_dir}")

def merge_incremental_data(jsonl_dir: Path, train_dir: Path):
    """Merge newly approved incremental data from WebAdmin into the JSONL training set."""
    incremental_dir = Path("/storage/exported/incremental")
    if not incremental_dir.exists():
        return
        
    tsv_dir = incremental_dir / "boxes_and_transcripts"
    inc_images_dir = incremental_dir / "images"
    if not tsv_dir.exists() or not inc_images_dir.exists():
        return
        
    print(f"➕ Merging incremental data from {incremental_dir}...")
    
    # 1. Copy incremental images to train_dir/imgs
    target_img_dir = train_dir / "imgs"
    for img_file in inc_images_dir.iterdir():
        if img_file.is_file():
            shutil.copy2(img_file, target_img_dir / img_file.name)
            
    # 2. Convert TSVs to JSONL format and append to train_df.jsonl
    jsonl_file = jsonl_dir / "train_df.jsonl"
    
    # Import cv2 or PIL to get image dimensions, we use PIL here since it's standard
    from PIL import Image
    
    records_added = 0
    with open(jsonl_file, "a", encoding="utf-8") as f_out:
        for tsv_file in tsv_dir.glob("*.tsv"):
            sample_id = tsv_file.stem
            # Find the corresponding image
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = target_img_dir / f"{sample_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
                    
            if not img_path:
                print(f"⚠️ Missing image for incremental sample {sample_id}, skipping.")
                continue
                
            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception as e:
                print(f"⚠️ Failed to read image size for {img_path}: {e}")
                continue
                
            words = []
            boxes = []
            labels = []
            
            with open(tsv_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip("\n").split("\t")
                    if len(parts) < 11:
                        continue
                    # format: index, x0, y0, x1, y1, x2, y2, x3, y3, text, label
                    x0 = float(parts[1])
                    y0 = float(parts[2])
                    x2 = float(parts[5])
                    y2 = float(parts[6])
                    
                    text = parts[9]
                    label = parts[10]
                    
                    x0_norm = max(0, min(1000, int(1000 * x0 / w_img)))
                    y0_norm = max(0, min(1000, int(1000 * y0 / h_img)))
                    x1_norm = max(0, min(1000, int(1000 * x2 / w_img)))
                    y1_norm = max(0, min(1000, int(1000 * y2 / h_img)))
                    
                    norm_box = [x0_norm, y0_norm, x1_norm, y1_norm]
                    
                    seg_words = text.split()
                    if not seg_words:
                        continue
                        
                    clean_label = label.strip()
                    if clean_label == "TIMESTAMPS":
                        clean_label = "TIMESTAMP"
                    elif clean_label == "TOTAL_TOTAL_COST":
                        clean_label = "TOTAL_COST"
                        
                    for idx, word in enumerate(seg_words):
                        words.append(word)
                        boxes.append(norm_box)
                        if clean_label == "O" or not clean_label:
                            labels.append("O")
                        else:
                            prefix = "B-" if idx == 0 else "I-"
                            labels.append(prefix + clean_label)
                            
            if words:
                record = {
                    "image": str(img_path),
                    "words": words,
                    "boxes": boxes,
                    "labels": labels
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_added += 1
                
    print(f"✅ Appended {records_added} incremental samples to JSONL training data.")

def build_label_maps(dataset: DatasetDict = None) -> Tuple[dict, dict]:
    """Generate ``label2id`` and ``id2label`` using a fixed, predefined label list to prevent index mismatches."""
    label_list = sorted(["O", "B-ADDRESS", "I-ADDRESS", "B-SELLER", "I-SELLER", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST"])
    label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return label2id, id2label

def preprocess_examples(examples, processor, label2id):
    """Tokenize a batch and align label ids with the tokenized output.
    The ``datasets`` library passes a dict of lists; we return the ``BatchEncoding``.
    """
    from PIL import Image
    # Map old absolute paths dynamically to the new consolidated DATA_ROOT
    mapped_images = []
    for img_path in examples["image"]:
        path_str = str(img_path).replace("\\", "/")
        if "bill_ocr/mc_ocr/layoutlmv3/data" in path_str:
            rel_path = path_str.split("layoutlmv3/data/")[-1]
            mapped_images.append(DATA_ROOT / rel_path)
        elif "layoutlmv3/data" in path_str:
            rel_path = path_str.split("layoutlmv3/data/")[-1]
            mapped_images.append(DATA_ROOT / rel_path)
        else:
            filename = Path(img_path).name
            found = False
            for sub in ["train/imgs", "val/imgs", "test/imgs"]:
                candidate = DATA_ROOT / sub / filename
                if candidate.is_file():
                    mapped_images.append(candidate)
                    found = True
                    break
            if not found:
                mapped_images.append(DATA_ROOT / filename)
                
    images = [Image.open(img).convert("RGB") for img in mapped_images]
    words = examples["words"]
    boxes = examples["boxes"]
    labels = examples["labels"]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    
    # Correctly align labels with subword tokens using word_ids
    batch_labels = []
    for batch_idx in range(len(words)):
        word_ids = encoding.word_ids(batch_index=batch_idx)
        label_seq = labels[batch_idx]
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx < len(label_seq):
                label_ids.append(label2id[label_seq[word_idx]])
            else:
                label_ids.append(-100)
        batch_labels.append(label_ids)
        
    encoding["labels"] = torch.tensor(batch_labels, dtype=torch.long)
    return encoding

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
from transformers import EarlyStoppingCallback, TrainerCallback

class ProgressCallback(TrainerCallback):
    def __init__(self, num_epochs, progress_file="/storage/layoutlmv3/training_progress.json"):
        self.num_epochs = num_epochs
        self.progress_file = Path(progress_file)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch = state.epoch or 0.0
            percent = min(100, int((epoch / self.num_epochs) * 100))
            
            data = {
                "isTraining": True,
                "stage": "training",
                "progress_percent": percent,
                "epoch": round(epoch, 2),
                "loss": round(logs["loss"], 4),
                "message": f"Đang huấn luyện (Epoch {int(epoch)}/{self.num_epochs}) - Loss: {logs['loss']:.4f}..."
            }
            try:
                self.progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.progress_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                pass

def train(num_epochs: int = 5, learning_rate: float = 2e-5, seed: int = 42, early_stop_patience: int = 3, resume_from: str | None = None):
    set_seed(seed)
    
    jsonl_dir = Path("/storage/layoutlmv3/jsonl")
    train_split_file = jsonl_dir / "train_split.jsonl"
    val_split_file = jsonl_dir / "val_split.jsonl"
    
    # 1️⃣ & 2️⃣ Convert / Load split datasets
    if resume_from and train_split_file.is_file() and val_split_file.is_file():
        print("🔄 Loading existing train/validation splits for resuming training...")
        copy_dataset(train=True)
        dataset = load_dataset("json", data_files={
            "train": str(train_split_file),
            "val": str(val_split_file)
        })
    else:
        # Clear output directory if not resuming
        out_dir = CHECKPOINT_PATH.parent
        if not resume_from and out_dir.exists():
            print(f"🧹 Clearing existing checkpoint directory: {out_dir}")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy training data
        copy_dataset(train=True)
        train_dir = DATA_ROOT / "train"
        
        # Convert CSV → JSONL
        convert_csv_to_jsonl(train_dir / "train_df.csv", jsonl_dir)
        
        # Merge Incremental Data
        merge_incremental_data(jsonl_dir, train_dir)
        
        # Load raw dataset to split it
        raw_dataset = load_dataset("json", data_files={"train": str(jsonl_dir / "train_df.jsonl")})
        
        # Split! (90% train, 10% val)
        split_ds = raw_dataset["train"].train_test_split(test_size=0.1, seed=seed)
        
        # Save splits back to disk for checkpointing
        print(f"💾 Saving train/val splits to {jsonl_dir}...")
        for split_name, split_data in [("train", split_ds["train"]), ("val", split_ds["test"])]:
            split_file = jsonl_dir / f"{split_name}_split.jsonl"
            with open(split_file, "w", encoding="utf-8") as f:
                for example in split_data:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    
        # Load split datasets
        dataset = load_dataset("json", data_files={
            "train": str(train_split_file),
            "val": str(val_split_file)
        })

    # 4️⃣ Build label maps
    label2id, id2label = build_label_maps(dataset)
    # 5️⃣ Processor & Model
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    if resume_from and (resume_from.endswith(".pth") or "model_best" in resume_from):
        checkpoint_path = Path(resume_from)
        if checkpoint_path.is_file():
            print(f"📥 Loading weights from custom checkpoint to start training: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            # Clear resume_from so trainer doesn't search for HF directory structure
            resume_from = None
    # 6️⃣ Tokenize & align labels
    def _preprocess(batch):
        return preprocess_examples(batch, processor, label2id)
    tokenized_ds = dataset.map(_preprocess, batched=True, remove_columns=dataset["train"].column_names)
    # 7️⃣ Trainer arguments
    args = TrainingArguments(
        output_dir=str(CHECKPOINT_PATH.parent),
        per_device_train_batch_size=4,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )
    # 8️⃣ Simple metric (F1) – used for early stopping
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        true_labels, true_preds = [], []
        for true, pred in zip(labels, preds):
            true_seq = [id2label[t] for t in true if t != -100]
            pred_seq = [id2label[p] for p, t in zip(pred, true) if t != -100]
            true_labels.append(true_seq)
            true_preds.append(pred_seq)
        return {"f1": f1_score(true_labels, true_preds)}

    train_dataset = tokenized_ds["train"]
    eval_dataset = tokenized_ds["val"]

    import inspect
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stop_patience), ProgressCallback(num_epochs)]
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
        
    trainer = Trainer(**trainer_kwargs)
    # 9️⃣ Train!
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    # 10️⃣ Save checkpoint (state_dict) and push to volume
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
    print(f"✅ Checkpoint saved to {CHECKPOINT_PATH}")
    VOLUME_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CHECKPOINT_PATH, VOLUME_CHECKPOINT)
    print(f"✅ Checkpoint copied to volume at {VOLUME_CHECKPOINT}")

# ---------------------------------------------------------------------------
def clean_single_tag(tag, token_str):
    return tag

# Evaluation (same split as training for demo)
# ---------------------------------------------------------------------------
def evaluate():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # Load dataset (must match training preprocessing)
    jsonl_dir = Path("/storage/layoutlmv3/jsonl")
    val_split_file = jsonl_dir / "val_split.jsonl"
    train_split_file = jsonl_dir / "train_split.jsonl"
    
    if train_split_file.is_file() and val_split_file.is_file():
        print(f"📊 Loading validation split from {val_split_file}...")
        dataset = load_dataset("json", data_files={
            "train": str(train_split_file),
            "val": str(val_split_file)
        })
        eval_split_name = "val"
    else:
        print(f"⚠️ Validation split not found, falling back to train_df.jsonl...")
        dataset = load_dataset("json", data_files={"train": str(jsonl_dir / "train_df.jsonl")})
        eval_split_name = "train"

    label2id, id2label = build_label_maps(dataset)
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    # Tokenize dataset for evaluation
    def _preprocess(batch):
        return preprocess_examples(batch, processor, label2id)
    tokenized_ds = dataset.map(_preprocess, batched=True, remove_columns=dataset[eval_split_name].column_names)
    tokenized_ds.set_format("torch")
    dl = DataLoader(tokenized_ds[eval_split_name], batch_size=4)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            preds = outputs.logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            input_ids_cpu = batch["input_ids"].cpu().numpy()
            
            for p, l, ids in zip(preds, labels, input_ids_cpu):
                true_seq = [id2label[t] for t in l if t != -100]
                
                # Apply rule-based post-processing on predictions
                pred_seq = []
                token_idx = 0
                for t, lt in zip(p, l):
                    if lt != -100:
                        raw_tag = id2label[t]
                        token_id = ids[token_idx]
                        token_str = processor.tokenizer.decode([token_id])
                        cleaned_tag = clean_single_tag(raw_tag, token_str)
                        pred_seq.append(cleaned_tag)
                    token_idx += 1
                
                all_labels.append(true_seq)
                all_preds.append(pred_seq)
                
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Evaluation metrics written to {RESULTS_PATH}\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")

# ---------------------------------------------------------------------------
# Test / inference – write predictions to CSV
# ---------------------------------------------------------------------------
def test():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # Copy test data (if volume provided)
    copy_dataset(train=False)
    test_dir = DATA_ROOT / "test"
    # Convert test CSV → JSONL
    jsonl_dir = DATA_ROOT / "jsonl_test"
    convert_csv_to_jsonl(test_dir / "test_df.csv", jsonl_dir)
    jsonl_file = jsonl_dir / "test_df.jsonl"
    if not jsonl_file.is_file() or jsonl_file.stat().st_size == 0:
        print("⚠️ Warning: Converted test JSONL is empty. Skipping test inference (requires pre-processed CSV with bounding boxes).")
        return
    # Load test dataset
    dataset = load_dataset("json", data_files={"test": str(jsonl_file)})
    
    # Load training dataset to build the correct label maps (must match training classes to prevent shape mismatch on loading checkpoint weights)
    train_split_file = Path("/storage/layoutlmv3/jsonl/train_split.jsonl")
    if train_split_file.is_file():
        print(f"📊 Loading label map from training split: {train_split_file}")
        train_dataset = load_dataset("json", data_files={"train": str(train_split_file)})
        label2id, id2label = build_label_maps(train_dataset)
    else:
        print("⚠️ Warning: train_split.jsonl not found. Falling back to default label map...")
        # Fallback to standard labels if train_split.jsonl is missing
        label_list = sorted(["O", "B-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST"])
        label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
        id2label = {idx: lbl for lbl, idx in label2id.items()}

    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    # Tokenize test set
    def _preprocess(batch):
        return preprocess_examples(batch, processor, label2id)
    # Remove all columns except "image" to preserve it for predictions mapping
    remove_cols = [c for c in dataset["test"].column_names if c != "image"]
    tokenized_ds = dataset.map(_preprocess, batched=True, remove_columns=remove_cols)
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "bbox", "pixel_values", "labels"])
    dl = DataLoader(tokenized_ds["test"], batch_size=4)
    rows = []
    global_idx = 0
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            preds = outputs.logits.argmax(-1).cpu().numpy()
            for idx, pred in enumerate(preds):
                img_path = tokenized_ds["test"][global_idx]["image"]
                global_idx += 1
                pred_tags = [id2label[t] for t, lt in zip(pred, batch["labels"][idx].numpy()) if lt != -100]
                entity_str = " ".join(pred_tags)
                rows.append({"image_path": img_path, "predicted_entities": entity_str})
    df = pd.DataFrame(rows)
    TEST_RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEST_RESULT_CSV, index=False)
    print(f"✅ Test predictions saved to {TEST_RESULT_CSV}\n{df.head()}")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LayoutLMv3 train / eval / test pipeline")
    parser.add_argument("action", choices=["train", "eval", "test"], help="Action to perform")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or file to resume training from")
    args = parser.parse_args()
    if args.action == "train":
        train(num_epochs=args.epochs, learning_rate=args.lr, seed=args.seed, resume_from=args.checkpoint)
    elif args.action == "eval":
        evaluate()
    else:
        test()
