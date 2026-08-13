"""
Modal Cloud Deployment Script for expense-ocr-nlu.
Provides serverless GPU execution for both API inference and model training.

Run local check/dev server:
  modal serve modal_app.py

Deploy serverless API:
  modal deploy modal_app.py

Run PICK KIE training:
  modal run modal_app.py::train_kie_model
"""
# Version 1.1.2 - Force rebuild
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import modal


# 1. Define Persistent Storage Volume on Modal
volume = modal.Volume.from_name("expense-ocr-nlu-storage", create_if_missing=True)

# 2. Define App
app = modal.App(name="expense-ocr-nlu")

# 3. Define the Docker Image with all KIE, OCR and NLU dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install C++ system packages required by OpenCV and PaddleOCR
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "libsm6", "libxext6", "libxrender-dev")
    # Install Pip requirements
    .pip_install(
        "fastapi==0.115.0",
        "google-genai",
        "uvicorn[standard]==0.30.6",
        "pydantic==2.9.2",
        "pydantic-settings==2.5.2",
        "python-multipart==0.0.9",
        "python-dotenv==1.0.1",
        "httpx==0.27.2",
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.45.0",
        "paddleocr==2.7.3",
        "paddlepaddle-gpu==2.6.1",  # GPU-accelerated PaddlePaddle
        "vietocr==0.3.13",
        "opencv-python-headless",  # Headless version for servers
        "pandas==2.2.2",
        "numpy==1.26.4",
        "joblib==1.4.2",
        "scikit-learn==1.5.2",
        "pyvi==0.1.1",
        "psycopg2-binary",
        "spacy==3.8.10",
        "overrides",
        "prefetch-generator",
        "torchtext==0.6.0",
        "Levenshtein",
        "yacs",
        "torchsummary",
        "datasets",
        "seqeval",
        "accelerate>=0.30.0",
        "peft>=0.12.0",
        "bitsandbytes",
        "einops",
        "sentencepiece",
        "triton",  # standard triton; provides GPU kernels on H100/A10
        "setuptools==69.5.1", # Added to fix 'pkg_resources' not found (pinned since v70+ might remove it)
    )
    # Pre-download PhoBERT and LayoutLMv3 weights during image building to avoid startup delays
    .run_commands(
        "python -c \""
        "from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForTokenClassification; "
        "AutoTokenizer.from_pretrained('vinai/phobert-base'); "
        "AutoModel.from_pretrained('vinai/phobert-base'); "
        "AutoProcessor.from_pretrained('microsoft/layoutlmv3-base'); "
        "AutoModelForTokenClassification.from_pretrained('microsoft/layoutlmv3-base'); "
        "\""
    )
    .add_local_dir(
        local_path=".",
        remote_path="/workspace",
        ignore=[".venv", "data", "exported", "__pycache__", ".git"],
        copy=True
    )
)



# ─── SECTION 1: INFERENCE WEB SERVER (FASTAPI) ───

# Dynamically construct secrets list to avoid hardcoding API keys in the codebase
secrets_list = []
# 1. Forward local environment variables loaded from .env during CLI deploy
local_keys = {}
for key in ["HF_TOKEN"]:
    val = os.environ.get(key)
    if val:
        local_keys[key] = val

if local_keys:
    secrets_list.append(modal.Secret.from_dict(local_keys))
else:
    # 2. Fall back to cloud-configured Modal Secret 'gemini-secrets' (kept for historical naming) if deploying from CI/CD
    secrets_list.append(modal.Secret.from_name("gemini-secrets"))



@app.function(
    image=image,
    volumes={"/storage": volume},  # Attach volume to fetch model weights
    gpu="l4",                     # Serverless GPU A10G or T4 for inference
    timeout=600,
    secrets=secrets_list,
    max_containers=5,
    min_containers=0,             # Scale to 0 when idle to save costs
    scaledown_window=900,  # Keep idle container warm for 15 minutes before shutting down
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    """ASGI entrypoint wrapping the unified FastAPI microservice."""
    import sys
    
    # Configure path references
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/src/api")
    sys.path.insert(0, "/workspace/text_nlu")

    # Set required environment variables
    os.environ["EXPENSE_OCR_NLU_DIR"] = "/workspace"
    os.environ["USE_REAL_NLU"] = "true"
    os.environ["USE_REAL_OCR"] = "true"
    os.environ["LAZY_LOAD_MODELS"] = "false"  # Load NLU immediately on container startup
    os.environ["PRELOAD_OCR"] = "true"        # Load OCR immediately on container startup
    os.environ["VERIFIED_OCR_LABELS_DIR"] = "/storage/exported"
    os.environ["USE_LOCAL_PHOGPT"] = "0"
    os.environ["USE_MODAL_PHOGPT"] = "1"
    os.environ["IS_MODAL"] = "true"

    # Deploy newly trained PICK KIE weights from volume if they exist
    volume_weights = Path("/storage/model_best.pth")
    if volume_weights.is_file():
        dest_weights = Path("/workspace/bill_ocr/models/pick_kie/model_best.pth")
        dest_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(volume_weights, dest_weights)
        print("[MODAL] Mounted newly trained PICK KIE weights from cloud volume.")

    # Deploy newly trained LayoutLMv3 weights from volume if they exist
    volume_layoutlmv3 = Path("/storage/layoutlmv3/model_best.pth")
    if volume_layoutlmv3.is_file():
        dest_layoutlmv3 = Path("/workspace/bill_ocr/models/layoutlmv3/model_best.pth")
        dest_layoutlmv3.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(volume_layoutlmv3, dest_layoutlmv3)
        print("[MODAL] Mounted newly trained LayoutLMv3 weights from cloud volume.")

    # Symlink newly trained Qwen model weights and LoRA adapter from volume if they exist
    volume_qwen = Path("/storage/qwen_vismimo")
    if volume_qwen.exists():
        dest_qwen = Path("/workspace/text_nlu/models/qwen_vismimo")
        dest_qwen.parent.mkdir(parents=True, exist_ok=True)
        if not dest_qwen.exists():
            os.symlink(volume_qwen, dest_qwen)
            print("[MODAL] Symlinked newly trained Qwen weights from cloud volume.")

    volume_qwen_lora = Path("/storage/qwen_vismimo_lora")
    if volume_qwen_lora.exists():
        dest_qwen_lora = Path("/workspace/text_nlu/models/qwen_vismimo_lora")
        dest_qwen_lora.parent.mkdir(parents=True, exist_ok=True)
        if not dest_qwen_lora.exists():
            os.symlink(volume_qwen_lora, dest_qwen_lora)
            print("[MODAL] Symlinked newly trained Qwen LoRA weights from cloud volume.")

    from app.main import create_app
    return create_app()


# ─── SECTION 2: LAYOUTLMV3 MODEL TRAINING FLOW ───

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="a10g",
    timeout=14400,
    secrets=secrets_list,
)
def train_layoutlmv3_model(num_epochs: int = 15, learning_rate: float = 5e-5, seed: int = 42, early_stop_patience: int = 3, resume_from_checkpoint: str = ""):
    """Train LayoutLMv3 model on cloud GPU and sync checkpoint to volume."""
    import sys
    import json
    import time
    from pathlib import Path
    from datetime import datetime, timezone

    sys.path.insert(0, "/workspace")
    from bill_ocr.layoutlmv3 import train_eval as layoutlmv3_cli
    
    t0 = time.time()
    layoutlmv3_cli.train(num_epochs=num_epochs, learning_rate=learning_rate, seed=seed, early_stop_patience=early_stop_patience, resume_from=resume_from_checkpoint)
    layoutlmv3_cli.evaluate()
    layoutlmv3_cli.test()
    duration = time.time() - t0

    # Save training metrics to volume history
    try:
        from bill_ocr.layoutlmv3.scripts import initialize_ocr_history
        initialize_ocr_history.main()  # Initialize history file if it does not exist

        metrics_file = Path("/storage/evaluation_metrics_layoutlmv3.txt")
        history_file = Path("/storage/layoutlmv3/ocr_training_history.json")

        if metrics_file.is_file():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            run_idx = len(history) + 1
            trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            p = metrics_data.get("precision", 0.0)
            r = metrics_data.get("recall", 0.0)
            f1 = metrics_data.get("f1", 0.0)

            history.append({
                "run_index": run_idx,
                "trained_at": trained_at,
                "duration_sec": round(duration, 2),
                "status": "success",
                "is_candidate": True,
                "metrics": {
                    "precision": round(p * 100, 2) if p <= 1.0 else p,
                    "recall": round(r * 100, 2) if r <= 1.0 else r,
                    "f1": round(f1 * 100, 2) if f1 <= 1.0 else f1,
                    "classification_report": metrics_data.get("classification_report", "")
                }
            })

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"💾 LayoutLMv3 Run #{run_idx} appended to training history.")
    except Exception as e:
        print(f"⚠️ Failed to append LayoutLMv3 run to history: {e}")
        
    # Reset training progress
    try:
        progress_file = Path("/storage/layoutlmv3/training_progress.json")
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({
                "isTraining": False,
                "stage": "done",
                "progress_percent": 100,
                "message": "Huấn luyện LayoutLMv3 hoàn tất!"
            }, f, ensure_ascii=False)
        print("✅ Reset training progress to done.")
    except Exception as e:
        print(f"⚠️ Failed to reset training progress: {e}")

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="l4",
    timeout=1800,
    secrets=secrets_list,
)
def evaluate_layoutlmv3_model():
    """Evaluate LayoutLMv3 model and write metrics to persistent volume."""
    import sys
    sys.path.insert(0, "/workspace")
    from bill_ocr.layoutlmv3 import train_eval as layoutlmv3_cli
    layoutlmv3_cli.evaluate()

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="l4",
    timeout=1800,
    secrets=secrets_list,
)
def test_layoutlmv3_model():
    """Run inference with LayoutLMv3 on test set and export results to CSV."""
    import sys
    sys.path.insert(0, "/workspace")
    from bill_ocr.layoutlmv3 import train_eval as layoutlmv3_cli
    layoutlmv3_cli.test()

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="l4",
    timeout=1800,
    secrets=secrets_list,
)
def visualize_layoutlmv3_test_predictions():
    """Run inference with LayoutLMv3 on 10 private test images and output visualized bounding boxes to volume."""
    import sys
    sys.path.insert(0, "/workspace")
    from bill_ocr.layoutlmv3.scripts import visualize_test_results
    visualize_test_results.main()


# ─── SECTION 4: PHOGPT LLM FINE-TUNING & INFERENCE SERVING ───

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="h100",
    timeout=14400,
    secrets=secrets_list,
)
def train_qwen_model(num_epochs: int = 3, learning_rate: float = 2e-4, batch_size: int = 4):
    """Fine-tune Qwen2.5-14B-Instruct model using LoRA on Nvidia H100 GPU."""
    import torch
    import json
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model
    from pathlib import Path
    import shutil

    model_id = "Qwen/Qwen2.5-14B-Instruct"
    
    # Authenticate with Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ Logged in to Hugging Face Hub.")
    else:
        print("⚠️ HF_TOKEN not set. Download may fail for gated repos.")

    print(f"📥 Loading base model and tokenizer: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load NLU dataset from text_nlu
    dataset_path = Path("/workspace/text_nlu/datasets/benchmark_finetune_2k.jsonl")
    if not dataset_path.exists():
        dataset_path = Path("text_nlu/datasets/benchmark_finetune_2k.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}!")
        
    print(f"📄 Loading dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    
    # Check if there is incremental data from web-admin
    inc_path = Path("/storage/exported/vistral_finetune_incremental.jsonl")
    if inc_path.is_file():
        print(f"➕ Merging incremental data from {inc_path}")
        inc_dataset = load_dataset("json", data_files=str(inc_path), split="train")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([dataset, inc_dataset])
        
    print(f"📊 Dataset loaded. Total samples: {len(dataset)}")
    
    # Split train/val (95% / 5%)
    split_ds = dataset.train_test_split(test_size=0.05, seed=42)
    
    def tokenize_func(example):
        return tokenizer(example["text"], truncation=True, max_length=512, padding=False)
        
    tokenized_dataset = split_ds.map(tokenize_func, batched=True, remove_columns=["text"])
    
    # Load model with 4-bit quantization (QLoRA) to save VRAM and prevent OOM
    print("🚀 Loading base model in 4-bit quantization (QLoRA)...")
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token
    )
    
    # Enable gradient checkpointing to drastically reduce memory usage during training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA config (targeting Qwen multi-head attention and MLP weights)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Output directory on persistent storage
    output_dir = Path("/storage/qwen_training_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=20,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )
    
    from transformers import TrainerCallback
    import time

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero and logs:
                progress_pct = round(state.global_step / state.max_steps * 100, 1) if state.max_steps > 0 else 0
                progress = {
                    "isTraining": True,
                    "stage": "TRAINING",
                    "progress_percent": progress_pct,
                    "message": f"Step {state.global_step}/{state.max_steps}",
                    "loss": logs.get("loss", 0),
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                    "updated_at": time.time()
                }
                progress_file = Path("/storage/llm_finetune/training_progress.json")
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False)

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                progress = {
                    "isTraining": False,
                    "stage": "SUCCESS",
                    "progress_percent": 100,
                    "message": "Huấn luyện LLM hoàn tất. Đang lưu mô hình...",
                    "updated_at": time.time()
                }
                progress_file = Path("/storage/llm_finetune/training_progress.json")
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False)
                    
    # Initialize Progress file
    init_progress = {
        "isTraining": True,
        "stage": "PREPARING",
        "progress_percent": 0,
        "message": "Đang chuẩn bị dữ liệu và mô hình Qwen...",
        "updated_at": time.time()
    }
    progress_file = Path("/storage/llm_finetune/training_progress.json")
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(init_progress, f, ensure_ascii=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback()]
    )
    
    print("🔥 Starting training...")
    trainer.train()
    
    # Save LoRA adapter weights
    lora_output_dir = Path("/storage/qwen_vismimo_lora")
    if lora_output_dir.exists():
        shutil.rmtree(lora_output_dir)
    lora_output_dir.mkdir(parents=True, exist_ok=True)
    print("💾 Saving LoRA adapter weights...")
    model.save_pretrained(str(lora_output_dir))

    # Merge LoRA weights into base model and save
    final_output_dir = Path("/storage/qwen_vismimo")
    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Merging LoRA weights and saving merged model locally to persistent volume...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    print(f"🎉 Success: Fine-tuned model saved to {final_output_dir}")

    # Save training logs/telemetry to /storage/llm_finetune/finetune_history.json
    try:
        import json
        from datetime import datetime, timezone
        
        history_points = []
        for log_entry in trainer.state.log_history:
            if "loss" in log_entry:
                history_points.append({
                    "step": log_entry.get("step"),
                    "epoch": round(float(log_entry.get("epoch")), 2),
                    "loss": round(float(log_entry.get("loss")), 4),
                    "learning_rate": float(log_entry.get("learning_rate", 0))
                })
        
        run_record = {
            "run_index": 1,
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "lora_target": "Maikhang/qwen-vismimo-lora",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_curve": history_points
        }
        
        history_dir = Path("/storage/llm_finetune")
        history_dir.mkdir(parents=True, exist_ok=True)
        
        history_file = history_dir / "finetune_history.json"
        
        # Load existing history if exists
        existing_history = []
        if history_file.is_file():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    existing_history = json.load(f)
            except Exception:
                pass
        
        # Determine next index
        next_idx = len(existing_history) + 1
        run_record["run_index"] = next_idx
        
        existing_history.append(run_record)
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(existing_history, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Saved fine-tune metrics to history: {history_file}")
    except Exception as e:
        print(f"⚠️ Failed to save training metrics: {e}")

    # Optionally push to Hugging Face Hub if HF_TOKEN is supplied
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("☁️ Hugging Face write token found. Commencing push to Hugging Face Hub...")
        try:
            from huggingface_hub import login, HfApi
            login(token=hf_token)
            
            # 1. Push LoRA adapter
            print("🚀 Pushing LoRA adapter to Maikhang/qwen-vismimo-lora...")
            api = HfApi()
            api.create_repo(repo_id="Maikhang/qwen-vismimo-lora", repo_type="model", exist_ok=True)
            model.push_to_hub("Maikhang/qwen-vismimo-lora", token=hf_token)
            tokenizer.push_to_hub("Maikhang/qwen-vismimo-lora", token=hf_token)
            
            # 2. Push Merged model
            print("🚀 Pushing merged model to Maikhang/qwen-vismimo...")
            api.create_repo(repo_id="Maikhang/qwen-vismimo", repo_type="model", exist_ok=True)
            merged_model.push_to_hub("Maikhang/qwen-vismimo", token=hf_token)
            tokenizer.push_to_hub("Maikhang/qwen-vismimo", token=hf_token)
            
            print("✨ Success: Pushed fine-tuned models to Hugging Face Hub repositories!")
        except Exception as e:
            print(f"⚠️ Failed to push to Hugging Face Hub: {e}")
    else:
        print("ℹ️ HF_TOKEN not found in environment. Skipping Hugging Face upload. Model remains stored on Modal persistent volume.")


@app.cls(
    image=image,
    volumes={"/storage": volume},
    gpu="a10g",
    timeout=600,
    secrets=secrets_list,
    max_containers=1,
    min_containers=0,                  # Scale to 0 when idle
    scaledown_window=900,  # Keep idle container alive for 15 minutes
    memory=32768,
)
class QwenModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from pathlib import Path
        from peft import PeftModel
        from transformers import AutoConfig
        
        merged_dir = Path("/storage/qwen_vismimo")
        if merged_dir.exists() and (merged_dir / "config.json").exists():
            local_model_dir = Path("/tmp/qwen_vismimo")
            if not local_model_dir.exists():
                print(f"⏳ Copying 28GB model from network volume to local SSD for fast loading... (takes ~1-2 mins)")
                import shutil
                shutil.copytree(merged_dir, local_model_dir)
            model_id = str(local_model_dir)
            print(f"🤖 Initializing fine-tuned merged Qwen model from local SSD: {model_id}")
        else:
            model_id = "Qwen/Qwen2.5-14B-Instruct"
            print(f"🤖 Merged model not found on volume. Initializing base model: {model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Quantize to 4-bit to fit comfortably on A10G (24GB VRAM) and maximize speed
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        if model_id == "Qwen/Qwen2.5-14B-Instruct":
            lora_dir = Path("/storage/qwen_vismimo_lora")
            if lora_dir.exists() and any(lora_dir.iterdir()):
                print(f"🤖 Applying fine-tuned LoRA adapter from: {lora_dir}")
                self.model = PeftModel.from_pretrained(base_model, str(lora_dir))
            else:
                print("🤖 LoRA adapter not found. Falling back to untuned base model.")
                self.model = base_model
        else:
            self.model = base_model
            
        self.model.eval()

    @modal.method()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch
        
        # Format using tokenizer's native chat template to avoid token mismatches
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print("Failed to apply tokenizer chat template, falling back to ChatML format:", e)
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding for absolute precision in JSON formatting
                tokenizer=self.tokenizer,
                stop_strings=["<|im_end|>", "<|im_start|>", "</s>", "Human:", "[INST]"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        
        # Log for debugging
        print(f"--- DEBUG QWEN PROMPT ---\n{prompt}\n---------------------------")
        print(f"--- DEBUG QWEN GENERATED ---\n{decoded}\n---------------------------")
        
        return decoded


@app.cls(
    image=image,
    volumes={"/storage": volume},
    gpu="a10g",
    timeout=600,
    secrets=secrets_list,
    max_containers=1,
    min_containers=0,
    scaledown_window=900,
    memory=32768,
)
class QwenBaseModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        model_id = "Qwen/Qwen2.5-14B-Instruct"
        print(f"🤖 Initializing STRICT BASE model: {model_id} (No LoRA)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model.eval()

    @modal.method()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                tokenizer=self.tokenizer,
                stop_strings=["<|im_end|>", "<|im_start|>", "</s>", "Human:", "[INST]"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return decoded


@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="l4",
    timeout=10800,
    secrets=secrets_list,
)
def train_nlu_model(target: str = "tfidf"):
    """Run NLU model retraining on cloud GPU and sync to storage."""
    import sys
    import json
    from pathlib import Path
    
    # Configure path references
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/src/api")
    
    # Set environment variables
    import os
    os.environ["EXPENSE_OCR_NLU_DIR"] = "/workspace"
    
    status_file = Path("/storage/nlu_models/training_status.json")
    try:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            json.dump({"training_active": True}, f)
        volume.commit()
        
        # Dynamically import and run the training flow from nlu router
        from app.routers.nlu import run_retraining
        run_retraining(Path("/workspace"), target)
        
    finally:
        with open(status_file, "w") as f:
            json.dump({"training_active": False}, f)
        volume.commit()
        print("🎉 NLU training process completed.")


@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="l4",
    timeout=10800,
    secrets=secrets_list,
)
def run_nlu_benchmark():
    """Run NLU benchmark across TF-IDF, PhoBERT and PhoGPT backends on cloud GPU."""
    import sys
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/src/api")
    sys.path.insert(0, "/workspace/text_nlu")
    
    from text_nlu.tools import run_nlu_benchmark
    run_nlu_benchmark.main()


@app.function(
    image=image,
    volumes={"/storage": volume},
    timeout=3600,
    secrets=secrets_list,
)
def sync_qwen_models_to_storage():
    """Tải đồng thời Base Model và Fine-tuned LoRA Model (nếu có trên HF) về /storage."""
    import os
    import shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ Cảnh báo: Không tìm thấy HF_TOKEN. Việc tải các mô hình (đặc biệt là repo cá nhân) có thể thất bại.")

    # 1. Tải Base Model
    base_model_id = "Qwen/Qwen2.5-14B-Instruct"
    base_dest_dir = Path("/storage/qwen_vismimo")
    print(f"📥 Đang tải Base Model [{base_model_id}] về {base_dest_dir}...")
    try:
        snapshot_download(
            repo_id=base_model_id,
            local_dir=str(base_dest_dir),
            token=hf_token,
            ignore_patterns=["*.pt", "*.bin"] # Chỉ ưu tiên safetensors
        )
        print("✅ Đã tải Base Model thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải Base Model: {e}")

    # 2. Tải Fine-tuned LoRA Model (Từ Hugging Face)
    lora_model_id = "Maikhang/qwen-vismimo-lora"
    lora_dest_dir = Path("/storage/qwen_vismimo_lora")
    print(f"📥 Đang thử tải LoRA Model [{lora_model_id}] về {lora_dest_dir}...")
    try:
        snapshot_download(
            repo_id=lora_model_id,
            local_dir=str(lora_dest_dir),
            token=hf_token
        )
        print("✅ Đã tải Fine-tuned LoRA Model thành công.")
    except Exception as e:
        print(f"⚠️ Không thể tải LoRA Model từ Hugging Face (Có thể bạn chưa push lên hoặc repo private): {e}")
        print("ℹ️ Quá trình train đang chạy trên Modal sẽ tự động tạo thư mục này sau khi hoàn tất.")

    volume.commit()
    print("🎉 Hoàn tất đồng bộ các mô hình Qwen vào /storage!")


@app.function(
    image=image,
    volumes={"/storage": volume},
    timeout=300,
)
def promote_layoutlmv3_model():
    """Promote candidate_model.pth to model_best.pth and backup current model."""
    import shutil
    import json
    from pathlib import Path
    
    candidate_path = Path("/storage/layoutlmv3/candidate_model.pth")
    best_path = Path("/storage/layoutlmv3/model_best.pth")
    previous_path = Path("/storage/layoutlmv3/model_previous.pth")
    
    if not candidate_path.is_file():
        return {"ok": False, "error": "No candidate model found."}
        
    # Backup current model if it exists
    if best_path.is_file():
        shutil.copy2(best_path, previous_path)
        
    # Promote candidate
    shutil.copy2(candidate_path, best_path)
    
    # Mark in history that it is no longer a candidate
    history_file = Path("/storage/layoutlmv3/ocr_training_history.json")
    if history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                
            for run in reversed(history):
                if run.get("is_candidate"):
                    run["is_candidate"] = False
                    break
                    
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error updating history: {e}")
            
    volume.commit()
    return {"ok": True, "message": "Model promoted successfully and old model backed up."}

@app.function(
    image=image,
    volumes={"/storage": volume},
    timeout=300,
)
def rollback_layoutlmv3_model():
    """Rollback model_best.pth to model_previous.pth."""
    import shutil
    import json
    from pathlib import Path
    
    best_path = Path("/storage/layoutlmv3/model_best.pth")
    previous_path = Path("/storage/layoutlmv3/model_previous.pth")
    
    if not previous_path.is_file():
        return {"ok": False, "error": "No previous model backup found."}
        
    # Overwrite best_path with previous_path
    shutil.copy2(previous_path, best_path)
    
    # Mark candidate status back? (Optional, just leave it as is, or we can just say rollback success)
    volume.commit()
    return {"ok": True, "message": "Model rolled back to previous version successfully."}
