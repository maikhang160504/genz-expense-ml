"""Local inference client for PhoGPT-7B5 running on Nvidia A10 (24GB VRAM) using Hugging Face."""
from __future__ import annotations

import os
import torch
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

_PHOGPT_MODEL = None
_PHOGPT_TOKENIZER = None
_MODEL_LOCK = threading.Lock()

def get_local_phogpt_path() -> str:
    """Resolve the path to the local Qwen-14B model, preferring fine-tuned weights if available."""
    from app.core.config import get_settings
    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    
    # Path to merged fine-tuned model
    ft_dir = nlu_dir / "text_nlu" / "models" / "qwen_vismimo"
    if ft_dir.exists() and any(ft_dir.iterdir()):
        logger.info("Found fine-tuned Qwen2.5-14B weights at: %s", ft_dir)
        return str(ft_dir)
    
    # Fallback to the untuned base model
    logger.info("No local fine-tuned Qwen weights found. Falling back to base HuggingFace model.")
    return "Qwen/Qwen2.5-14B-Instruct"

def load_local_phogpt():
    """Load the model and tokenizer into VRAM once (Singleton). Uses 4-bit quantization for GPU."""
    global _PHOGPT_MODEL, _PHOGPT_TOKENIZER
    if _PHOGPT_MODEL is not None:
        return
        
    with _MODEL_LOCK:
        if _PHOGPT_MODEL is not None:
            return
        
    diag_msgs = []
    try:
        import accelerate
        diag_msgs.append(f"accelerate: ok {accelerate.__version__}")
    except Exception as e:
        import traceback
        diag_msgs.append(f"accelerate: failed\n{traceback.format_exc()}")

    try:
        import bitsandbytes as bnb
        diag_msgs.append(f"bitsandbytes: ok {bnb.__version__}")
    except Exception as e:
        import traceback
        diag_msgs.append(f"bitsandbytes: failed\n{traceback.format_exc()}")
        
    try:
        import torch
        diag_msgs.append(f"torch: ok {torch.__version__}")
        diag_msgs.append(f"cuda_available: {torch.cuda.is_available()}")
        diag_msgs.append(f"device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            diag_msgs.append(f"current_device: {torch.cuda.current_device()}")
            diag_msgs.append(f"device_name: {torch.cuda.get_device_name(0)}")
        else:
            # Let's run nvidia-smi
            import subprocess
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            diag_msgs.append(f"nvidia-smi return code: {res.returncode}")
            diag_msgs.append(f"nvidia-smi stdout:\n{res.stdout}")
            diag_msgs.append(f"nvidia-smi stderr:\n{res.stderr}")
            
            # Let's log CUDA environment variables
            import os
            cuda_envs = {k: v for k, v in os.environ.items() if "CUDA" in k or "LD_LIBRARY" in k}
            diag_msgs.append(f"CUDA/LD envs: {cuda_envs}")
    except Exception as e:
        import traceback
        diag_msgs.append(f"torch/cuda diagnostic failed\n{traceback.format_exc()}")
        
    try:
        with open("/storage/diagnostic_llm.log", "w") as f:
            f.write("\n---\n".join(diag_msgs))
    except Exception as ex:
        print("Failed to write diagnostic file:", ex)
        
    model_path = get_local_phogpt_path()
    logger.info("Initializing local Qwen2.5-14B model from: %s", model_path)
    
    _PHOGPT_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    if _PHOGPT_TOKENIZER.pad_token is None:
        _PHOGPT_TOKENIZER.pad_token = _PHOGPT_TOKENIZER.eos_token
        
    # Standard 4-bit quantization config to fit on GPU
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Check for local LoRA adapter weights
    from peft import PeftModel
    from app.core.config import get_settings
    settings = get_settings()
    nlu_dir = Path(settings.expense_ocr_nlu_dir).resolve()
    lora_dir = nlu_dir / "text_nlu" / "models" / "qwen_vismimo_lora"
    if lora_dir.exists() and any(lora_dir.iterdir()):
        logger.info("Applying local fine-tuned LoRA adapter from: %s", lora_dir)
        _PHOGPT_MODEL = PeftModel.from_pretrained(base_model, str(lora_dir))
    else:
        _PHOGPT_MODEL = base_model
        
    _PHOGPT_MODEL.eval()
    logger.info("Local Qwen2.5-14B initialized successfully on GPU.")

def run_local_phogpt_inference(system_prompt: str, user_prompt: str) -> str:
    """Run local inference using the cached Qwen2.5-14B model."""
    global _PHOGPT_MODEL, _PHOGPT_TOKENIZER
    if _PHOGPT_MODEL is None:
        load_local_phogpt()
        
    # Format using tokenizer's native chat template to avoid token mismatches
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = _PHOGPT_TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.warning("Failed to apply tokenizer chat template, falling back to ChatML format: %s", e)
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = _PHOGPT_TOKENIZER(prompt, return_tensors="pt").to(_PHOGPT_MODEL.device)
    with torch.no_grad():
        outputs = _PHOGPT_MODEL.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,  # Greedy decoding for absolute precision
            tokenizer=_PHOGPT_TOKENIZER,
            stop_strings=["<|im_end|>", "<|im_start|>", "</s>", "Human:", "[INST]"],
            eos_token_id=_PHOGPT_TOKENIZER.eos_token_id,
            pad_token_id=_PHOGPT_TOKENIZER.eos_token_id,
        )
        
    input_len = inputs["input_ids"].shape[1]
    decoded = _PHOGPT_TOKENIZER.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return decoded
