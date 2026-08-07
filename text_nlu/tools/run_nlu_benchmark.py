"""NLU Backend Benchmark Evaluator.

Runs a test suite of Vietnamese financial sentences across:
1) TF-IDF + SVM/Logistic Regression (Production)
2) PhoBERT Encoder
3) Qwen 2.5 LoRA (LLM)

Computes F1-score, Accuracy, and average Latency for the Thesis.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
import sys
import io

# Fix terminal stdout encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

import numpy as np
from sklearn.metrics import f1_score

def ensure_benchmark_dataset(dataset_path: Path):
    if dataset_path.is_file():
        print(f"✅ Found existing benchmark dataset at {dataset_path}, using it directly without regenerating.")
        return
        
    print(f"🛠️ Creating 100-sample NLU Benchmark Dataset from real data at {dataset_path}...")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    import csv
    import random
    from collections import defaultdict
    
    # Use fixed seed for reproducibility
    random.seed(42)
    samples = []
    
    # 1. Load Records (Diverse by Category)
    record_path = ROOT / "text_nlu" / "datasets" / "intent_record.csv"
    if record_path.is_file():
        records_by_cat = defaultdict(list)
        with open(record_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[0].strip() and row[1].strip():
                    cat = row[1].strip()
                    rec_type = "income" if row[2].strip().lower() == "income" else "expense"
                    records_by_cat[cat].append({
                        "text": row[0].strip(),
                        "expected_intent": "Record",
                        "expected_category": cat,
                        "expected_record_type": rec_type,
                        "expected_action_type": "None"
                    })
        # Sample evenly across categories (total ~45)
        categories = list(records_by_cat.keys())
        random.shuffle(categories)
        records_sampled = 0
        while records_sampled < 45 and any(records_by_cat.values()):
            for cat in categories:
                if records_by_cat[cat] and records_sampled < 45:
                    item = random.choice(records_by_cat[cat])
                    samples.append(item)
                    records_by_cat[cat].remove(item)
                    records_sampled += 1
                        
    # 2. Load Actions (Diverse by Action Type)
    action_path = ROOT / "text_nlu" / "datasets" / "intent_action.csv"
    if action_path.is_file():
        actions_by_type = defaultdict(list)
        with open(action_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 5 and row[0].strip() and row[1].strip() == "Action":
                    act_type = row[2].strip()
                    cat = row[4].strip() if len(row) > 4 and row[4].strip() else "None"
                    actions_by_type[act_type].append({
                        "text": row[0].strip(),
                        "expected_intent": "Action",
                        "expected_category": cat,
                        "expected_record_type": "None",
                        "expected_action_type": act_type
                    })
        # Sample evenly across action types (total ~35)
        act_types = list(actions_by_type.keys())
        random.shuffle(act_types)
        actions_sampled = 0
        while actions_sampled < 35 and any(actions_by_type.values()):
            for act in act_types:
                if actions_by_type[act] and actions_sampled < 35:
                    item = random.choice(actions_by_type[act])
                    samples.append(item)
                    actions_by_type[act].remove(item)
                    actions_sampled += 1

    # 3. Load Chitchat
    chitchat_path = ROOT / "text_nlu" / "datasets" / "intent_chitchat.csv"
    if chitchat_path.is_file():
        all_chitchats = []
        with open(chitchat_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    all_chitchats.append({
                        "text": row[0].strip(),
                        "expected_intent": "Chitchat",
                        "expected_category": "None",
                        "expected_record_type": "None",
                        "expected_action_type": "None"
                    })
        if all_chitchats:
            chitchats_sampled = random.sample(all_chitchats, min(20, len(all_chitchats)))
            samples.extend(chitchats_sampled)
            
    # Shuffle the final dataset so it's a mix
    random.shuffle(samples)
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("✅ Diverse NLU Benchmark Dataset generated successfully.")

def calc_metrics(y_true, y_pred):
    if not y_true: return 0.0, 0.0
    y_true_lower = [str(t).strip().lower() for t in y_true]
    y_pred_lower = [str(p).strip().lower() for p in y_pred]
    acc = sum(1 for yt, yp in zip(y_true_lower, y_pred_lower) if yt == yp) / len(y_true_lower) * 100
    f1 = f1_score(y_true_lower, y_pred_lower, average='macro', zero_division=0) * 100
    return acc, f1

def evaluate_backend(name, samples, backend_override="tfidf") -> dict:
    print(f"⏳ Evaluating {name} backend...")
    
    os.environ["NLU_USE_ENCODER"] = "1" if backend_override == "encoder" else "0"
    if backend_override == "llm":
        os.environ["IS_MODAL"] = "true"
    
    # Monkey-patch the models.py so it forces the correct backend for benchmark isolation
    import src.nlu.models
    src.nlu.models.get_intent_backend = lambda: backend_override
    src.nlu.models.get_category_backend = lambda: backend_override
    
    from src.nlu.models import load_intent_model, load_category_model, load_action_type_model, load_record_type_model, load_chitchat_sentiment_model, load_action_slots_model
    from src.nlu.ner import load_ner_model
    from src.nlu.pipeline import run_nlu
    from src.config import settings
    
    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    slots_m = load_action_slots_model()
    rec_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    ner = load_ner_model(settings.NER_MODEL_DIR)
    
    y_true_intent, y_pred_intent = [], []
    y_true_cat, y_pred_cat = [], []
    y_true_rec, y_pred_rec = [], []
    y_true_act, y_pred_act = [], []
    latencies = []
    
    mismatches = []
    
    for s in samples:
        t0 = time.time()
        res = run_nlu(s["text"], intent_m, cat_m, act_m, rec_m, sent_m, ner, slots_m)
        latencies.append((time.time() - t0) * 1000)
        
        def norm_cat(c):
            c_str = str(c or "None").strip().lower()
            if c_str in ("null", "none", "", "nil"):
                return "None"
            return str(c).strip()
            
        pred_intent = str(res.get("intent") or "Unknown")
        pred_cat = norm_cat(res.get("category"))
        pred_rec = str(res.get("record_type") or "None")
        pred_act = str(res.get("action_type") or "None")
        
        y_true_intent.append(s["expected_intent"])
        y_pred_intent.append(pred_intent)
        if pred_intent.lower() != str(s["expected_intent"]).lower():
            mismatches.append(f"  - [Intent] Text: '{s['text']}' | True: '{s['expected_intent']}' | Pred: '{pred_intent}'")

        exp_intent = s.get("expected_intent")

        # Record type evaluated strictly on Record intent
        if exp_intent == "Record":
            exp_rec = s.get("expected_rec_type", s.get("expected_record_type", "expense"))
            y_true_rec.append(exp_rec)
            y_pred_rec.append(pred_rec)

        # Category evaluated on Record OR specific Action types that need it
        ACTIONS_WITH_CATEGORY = {"REPORT_GENERAL", "REPORT_COMPARE", "SET_LIMIT", "SEARCH_RECORD", "SUGGEST_BUDGET"}
        if exp_intent == "Record" or (exp_intent == "Action" and s.get("expected_action_type") in ACTIONS_WITH_CATEGORY):
            exp_cat = norm_cat(s.get("expected_category"))
            y_true_cat.append(exp_cat)
            y_pred_cat.append(pred_cat)
            if pred_cat.lower() != exp_cat.lower():
                mismatches.append(f"  - [Cat] Text: '{s['text']}' | True: '{s.get('expected_category')}' | Pred: '{pred_cat}'")

        # Action type evaluated strictly on Action intent
        if exp_intent == "Action":
            exp_act = s.get("expected_action_type", "None")
            y_true_act.append(exp_act)
            y_pred_act.append(pred_act)
            if pred_act.lower() != str(exp_act).lower():
                mismatches.append(f"  - [Act] Text: '{s['text']}' | True: '{exp_act}' | Pred: '{pred_act}'")
            
    os.environ["NLU_USE_ENCODER"] = "0"
    
    intent_acc, intent_f1 = calc_metrics(y_true_intent, y_pred_intent)
    category_acc, category_f1 = calc_metrics(y_true_cat, y_pred_cat)
    record_acc, record_f1 = calc_metrics(y_true_rec, y_pred_rec)
    action_acc, action_f1 = calc_metrics(y_true_act, y_pred_act)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    
    if mismatches:
        print(f"\n[!] Mismatches for {name}:")
        for m in mismatches[:10]:
            print(m)
            
    return {
        "intent_accuracy": round(intent_acc, 2), "intent_f1": round(intent_f1, 2),
        "category_accuracy": round(category_acc, 2), "category_f1": round(category_f1, 2),
        "record_type_accuracy": round(record_acc, 2), "record_type_f1": round(record_f1, 2),
        "action_type_accuracy": round(action_acc, 2), "action_type_f1": round(action_f1, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2)
    }

def main():
    if os.path.exists("/workspace"):
        dataset_path = Path("/workspace/text_nlu/datasets/nlu_benchmark.jsonl")
    else:
        dataset_path = ROOT / "text_nlu" / "datasets" / "nlu_benchmark.jsonl"
        
    ensure_benchmark_dataset(dataset_path)
    
    # Read samples
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    print(f"✅ Loaded {len(samples)} benchmark samples from the original dataset.")
    # No capping - use the full dataset provided by the user
    
    results = {}
    results["tfidf"] = evaluate_backend("TF-IDF", samples, backend_override="tfidf")
    results["phobert"] = evaluate_backend("PhoBERT", samples, backend_override="encoder")
    results["qwen25_lora"] = evaluate_backend("Qwen 2.5-14B LoRA Fine-tuned", samples, backend_override="llm")
        
    # Save output
    if os.path.exists("/storage"):
        output_file = Path("/storage/nlu_models/nlu_benchmark_results.json")
    else:
        output_file = ROOT / "text_nlu" / "datasets" / "nlu_benchmark_results.json"
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 Benchmark successfully completed! Results written to: {output_file}")
    
    print("\n" + "="*145)
    print("                                      BẢNG SO SÁNH HIỆU NĂNG NLU ĐÁNH GIÁ LUẬN VĂN")
    print("="*145)
    print(f"{'Mô hình NLU':<15} | {'Intent (Acc/F1)':<18} | {'Action (Acc/F1)':<18} | {'Category (Acc/F1)':<18} | {'Record (Acc/F1)':<18} | {'Avg Latency':<12} | {'P95 Latency':<12}")
    print("-" * 145)
    for model_key, res in results.items():
        name = "TF-IDF" if model_key == "tfidf" else "PhoBERT" if model_key == "phobert" else "Qwen 2.5"
        intent_str = f"{res['intent_accuracy']:.1f}%/{res['intent_f1']:.1f}%"
        if model_key == "qwen25_lora":
            action_str = f"{res['action_type_accuracy']:.1f}%/{res['action_type_f1']:.1f}%"
        else:
            action_str = "N/A"
        cat_str = f"{res['category_accuracy']:.1f}%/{res['category_f1']:.1f}%"
        rec_str = f"{res['record_type_accuracy']:.1f}%/{res['record_type_f1']:.1f}%"
        print(f"{name:<15} | {intent_str:<18} | {action_str:<18} | {cat_str:<18} | {rec_str:<18} | {res['avg_latency_ms']:<9.1f} ms | {res['p95_latency_ms']:<9.1f} ms")
    print("="*145)

if __name__ == "__main__":
    main()
