import json
import time
import numpy as np
import modal
from pathlib import Path
from unittest.mock import patch
from sklearn.metrics import accuracy_score, f1_score

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.nlu.llm_intent_handler as handler

INPUT_FILE = ROOT / "text_nlu" / "datasets" / "nlu_benchmark.json"

print("🔄 Connecting to Modal Endpoints...")
QwenBaseModel = modal.Cls.from_name("expense-ocr-nlu", "QwenBaseModel")
base_client = QwenBaseModel()

QwenModel = modal.Cls.from_name("expense-ocr-nlu", "QwenModel")
lora_client = QwenModel()

def call_base_llm(system_prompt, user_prompt, **kwargs):
    return base_client.generate.remote(system_prompt, user_prompt)

def call_lora_llm(system_prompt, user_prompt, **kwargs):
    return lora_client.generate.remote(system_prompt, user_prompt)

def evaluate_model(samples, model_caller_fn):
    true_intents = []
    pred_intents = []
    
    true_categories = []
    pred_categories = []
    
    exact_match_count = 0
    valid_json_count = 0
    
    latencies = []
    
    for idx, sample in enumerate(samples):
        text = sample["text"]
        gt_intent = sample.get("intent", "Chitchat")
        gt_category = sample.get("category", "")
        if not gt_category: gt_category = "None"
        gt_amount = sample.get("amount", 0)
        gt_action = sample.get("action_type", "None")
        if not gt_action: gt_action = "None"
        
        t0 = time.time()
        
        with patch('src.nlu.llm_intent_handler._call_llm', side_effect=model_caller_fn):
            # run end-to-end NLU with Qwen
            result = handler.run_llm_nlu_v2(text)
            
        latency = time.time() - t0
        latencies.append(latency)
        
        pred_intent = result.get("intent", "Chitchat")
        pred_category = result.get("category", "")
        if not pred_category: pred_category = "None"
        pred_amount = result.get("amount", 0)
        
        # Determine exact match based on Intent
        is_exact = False
        if gt_intent == pred_intent:
            if gt_intent == "Record":
                if str(gt_category) == str(pred_category) and str(gt_amount) == str(pred_amount):
                    is_exact = True
            elif gt_intent == "Action":
                p_action = result.get("action_type", "None")
                if not p_action: p_action = "None"
                if str(gt_action) == str(p_action):
                    is_exact = True
            else: # Chitchat
                is_exact = True
                
        if is_exact:
            exact_match_count += 1
            
        true_intents.append(gt_intent)
        pred_intents.append(pred_intent)
        
        if gt_intent == "Record":
            true_categories.append(gt_category)
            pred_categories.append(pred_category)
            
        # If llm_json exists, we assume valid json was returned by stage 2
        if result.get("llm_json") is not None and isinstance(result.get("llm_json"), dict) and len(result.get("llm_json")) > 0:
            valid_json_count += 1
            
        print(f"[{idx+1}/{len(samples)}] Text: {text} | GT: {gt_intent}, {gt_category} | Pred: {pred_intent}, {pred_category} | Exact: {is_exact}")
            
    # Calculate metrics
    acc = accuracy_score(true_intents, pred_intents)
    
    # We combine Intent and Category to calculate a holistic F1 (or just average them)
    # Let's calculate F1 for Intents
    intent_f1 = f1_score(true_intents, pred_intents, average='macro', zero_division=0)
    
    if len(true_categories) > 0:
        cat_f1 = f1_score(true_categories, pred_categories, average='macro', zero_division=0)
    else:
        cat_f1 = 1.0
        
    avg_f1 = (intent_f1 + cat_f1) / 2
    
    exact_match_rate = exact_match_count / len(samples)
    json_valid_rate = valid_json_count / len(samples)
    avg_latency = np.mean(latencies)
    
    return {
        "accuracy": acc * 100,
        "f1": avg_f1 * 100,
        "exact_match": exact_match_rate * 100,
        "json_valid": json_valid_rate * 100,
        "latency": avg_latency
    }


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    print(f"Loaded {len(samples)} test cases.")
    
    print("\n=============================================")
    print("🚀 RUNNING QWEN 2.5 BASE MODEL EVALUATION")
    print("=============================================")
    base_metrics = evaluate_model(samples, call_base_llm)
    
    print("\n=============================================")
    print("🚀 RUNNING QWEN 2.5 + LORA MODEL EVALUATION")
    print("=============================================")
    lora_metrics = evaluate_model(samples, call_lora_llm)
    
    print("\n\n📊 FINAL BENCHMARK RESULTS")
    print("-" * 110)
    print(f"| {'Model':<20} | {'Accuracy':<12} | {'F1-score':<12} | {'Exact Match':<15} | {'JSON Valid Rate':<17} | {'Latency':<10} |")
    print("-" * 110)
    
    def fmt_row(name, m):
        return f"| {name:<20} | {m['accuracy']:>8.1f}%   | {m['f1']:>8.1f}%   | {m['exact_match']:>11.1f}%   | {m['json_valid']:>13.1f}%   | {m['latency']:>6.1f} s   |"
        
    print(fmt_row("Qwen 2.5 Base", base_metrics))
    print(fmt_row("Qwen 2.5 + LoRA", lora_metrics))
    print("-" * 110)
    
    
if __name__ == "__main__":
    main()
