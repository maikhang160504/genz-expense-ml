import json
import os
import time
from pathlib import Path
import sys
import io
import numpy as np
from sklearn.metrics import f1_score
from unittest.mock import patch

# Fix terminal stdout encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

# Load Modal only when evaluating Qwen
modal_loaded = False
base_client = None
lora_client = None

def init_modal():
    global modal_loaded, base_client, lora_client
    if not modal_loaded:
        print("🔄 Connecting to Modal Endpoints...")
        try:
            from modal_app import QwenBaseModel, QwenModel
            base_client = QwenBaseModel()
            lora_client = QwenModel()
        except ImportError:
            import modal
            QwenBaseModel = modal.Cls.from_name("expense-ocr-nlu", "QwenBaseModel")
            base_client = QwenBaseModel()
            QwenModel = modal.Cls.from_name("expense-ocr-nlu", "QwenModel")
            lora_client = QwenModel()
        modal_loaded = True

def call_base_llm(system_prompt, user_prompt, **kwargs):
    init_modal()
    return base_client.generate.remote(system_prompt, user_prompt)

def call_lora_llm(system_prompt, user_prompt, **kwargs):
    init_modal()
    return lora_client.generate.remote(system_prompt, user_prompt)

def calc_metrics(y_true, y_pred):
    if not y_true: return 0.0, 0.0
    y_true_lower = [str(t).strip().lower() for t in y_true]
    y_pred_lower = [str(p).strip().lower() for p in y_pred]
    acc = sum(1 for yt, yp in zip(y_true_lower, y_pred_lower) if yt == yp) / len(y_true_lower) * 100
    
    # Bỏ qua nhãn 'none', 'null' hoặc chuỗi rỗng khi tính F1-Macro
    valid_labels = list(set([t for t in y_true_lower + y_pred_lower if t not in ["", "none", "null"]]))
    
    if not valid_labels:
        f1 = acc
    else:
        f1 = f1_score(y_true_lower, y_pred_lower, labels=valid_labels, average='macro', zero_division=0) * 100
        
    return acc, f1

def evaluate_backend(name, samples, backend_override="tfidf") -> dict:
    print(f"\n⏳ Evaluating {name}...")
    
    # Environment variables & Mock Target
    if backend_override in ("tfidf", "encoder"):
        os.environ["NLU_USE_ENCODER"] = "1" if backend_override == "encoder" else "0"
        os.environ["IS_MODAL"] = "false"
        mock_target = None
    elif backend_override == "qwen_base":
        os.environ["IS_MODAL"] = "true"
        mock_target = call_base_llm
    elif backend_override == "qwen_lora":
        os.environ["IS_MODAL"] = "true"
        mock_target = call_lora_llm

    # Patch for TF-IDF / PhoBERT models loading
    import src.nlu.models
    orig_intent = src.nlu.models.get_intent_backend
    orig_cat = src.nlu.models.get_category_backend
    
    if backend_override in ("tfidf", "encoder"):
        src.nlu.models.get_intent_backend = lambda: backend_override
        src.nlu.models.get_category_backend = lambda: backend_override

    from src.nlu.models import load_intent_model, load_category_model, load_action_type_model, load_record_type_model, load_chitchat_sentiment_model, load_action_slots_model
    from src.nlu.ner import load_ner_model
    from src.nlu.pipeline import run_nlu, classify_intent
    from src.nlu.llm_intent_handler import run_llm_nlu_v2, classify_intent_llm
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
    intent_l = []
    cat_l = []
    
    mismatches = []
    
    for idx, s in enumerate(samples):
        exp_intent = s.get("intent", "Chitchat")
        exp_rec = s.get("record_type") or "None"
        exp_act = s.get("action_type") or "None"
        slots = s.get("slots") or {}
        exp_cat = slots.get("category") or "None"
        
        # Inference
        if backend_override in ("tfidf", "encoder"):
            t0 = time.time()
            res = run_nlu(s["text"], intent_m, cat_m, act_m, rec_m, sent_m, ner, slots_m)
            total_time = (time.time() - t0) * 1000
            latencies.append(total_time)
            
            # Stage 1: Intent Latency
            ti = time.time()
            classify_intent(s["text"], intent_m)
            intent_l.append((time.time() - ti) * 1000)
            
            # Stage 2: Category Latency (Only meaningful for Record intent in ML pipeline)
            if exp_intent == "Record":
                tc = time.time()
                if cat_m.get("backend") == "encoder" and cat_m.get("bundle"):
                    from src.nlu.encoder_runtime import predict_category_encoder
                    predict_category_encoder(cat_m["bundle"], s["text"])
                elif cat_m.get("backend") == "tfidf" and cat_m.get("vectorizer"):
                    from pipeline.text_preprocessing import clean_category_text
                    cat_vec = cat_m["vectorizer"].transform([clean_category_text(s["text"])])
                    cat_m["model"].predict(cat_vec)
                cat_l.append((time.time() - tc) * 1000)
                
        else: # Qwen Base or LoRA
            t0 = time.time()
            with patch('src.nlu.llm_intent_handler._call_llm', side_effect=mock_target):
                # Stage 1
                t1 = time.time()
                pred_intent, _ = classify_intent_llm(s["text"])
                intent_l.append((time.time() - t1) * 1000)
                
                # Stage 2
                t2 = time.time()
                res = run_llm_nlu_v2(s["text"], forced_intent=pred_intent)
                cat_l.append((time.time() - t2) * 1000)
                
            total_time = (time.time() - t0) * 1000
            latencies.append(total_time)
            
        def norm_cat(c):
            c_str = str(c or "None").strip().lower()
            if c_str in ("null", "none", "", "nil"):
                return "None"
            return str(c).strip()
            
        pred_intent = str(res.get("intent") or "Unknown")
        pred_rec = str(res.get("record_type") or "None")
        pred_act = str(res.get("action_type") or "None")
        
        # Robust category extraction (especially for Action intent in TF-IDF/PhoBERT)
        pred_cat_val = res.get("category")
        if not pred_cat_val and pred_intent == "Action":
            act_details = res.get("action_details") or {}
            # Action category is often mapped to "target" internally
            pred_cat_val = act_details.get("target") or act_details.get("category_code") or act_details.get("category")
            
        pred_cat = norm_cat(pred_cat_val)
        
        # 1. Intent Accuracy
        y_true_intent.append(exp_intent)
        y_pred_intent.append(pred_intent)
        
        if pred_intent.lower() != str(exp_intent).lower():
            mismatches.append(f"  - [Intent] Text: '{s['text']}' | True: '{exp_intent}' | Pred: '{pred_intent}'")

        # 2. Record Type
        if exp_intent == "Record" or pred_intent == "Record":
            y_true_rec.append(exp_rec if exp_intent == "Record" else "None")
            y_pred_rec.append(pred_rec if pred_intent == "Record" else "None")

        # 3. Category
        ACTIONS_WITH_CATEGORY = {"REPORT_GENERAL", "REPORT_COMPARE", "SET_LIMIT", "SEARCH_RECORD", "SUGGEST_BUDGET"}
        exp_has_cat = exp_intent == "Record" or (exp_intent == "Action" and exp_act in ACTIONS_WITH_CATEGORY)
        pred_has_cat = pred_intent == "Record" or (pred_intent == "Action" and pred_act in ACTIONS_WITH_CATEGORY)
        
        if exp_has_cat or pred_has_cat:
            y_true_cat.append(norm_cat(exp_cat) if exp_has_cat else "None")
            y_pred_cat.append(pred_cat if pred_has_cat else "None")
            
            if pred_has_cat and exp_has_cat and pred_cat != norm_cat(exp_cat):
                mismatches.append(f"  - [Cat] Text: '{s['text']}' | True: '{norm_cat(exp_cat)}' | Pred: '{pred_cat}'")

        # 4. Action Type
        if exp_intent == "Action" or pred_intent == "Action":
            y_true_act.append(exp_act if exp_intent == "Action" else "None")
            y_pred_act.append(pred_act if pred_intent == "Action" else "None")
                
    # Restore monkey-patch
    src.nlu.models.get_intent_backend = orig_intent
    src.nlu.models.get_category_backend = orig_cat
            
    intent_acc, intent_f1 = calc_metrics(y_true_intent, y_pred_intent)
    category_acc, category_f1 = calc_metrics(y_true_cat, y_pred_cat)
    record_acc, record_f1 = calc_metrics(y_true_rec, y_pred_rec)
    action_acc, action_f1 = calc_metrics(y_true_act, y_pred_act)
    
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_intent_latency = float(np.mean(intent_l)) if intent_l else 0.0
    avg_cat_latency = float(np.mean(cat_l)) if cat_l else 0.0
    
    if mismatches:
        print(f"\n[!] Top 5 Mismatches for {name}:")
        for m in mismatches[:5]:
            print(m)
            
    return {
        "intent_accuracy": round(intent_acc, 2), "intent_f1": round(intent_f1, 2),
        "category_accuracy": round(category_acc, 2), "category_f1": round(category_f1, 2),
        "record_type_accuracy": round(record_acc, 2), "record_type_f1": round(record_f1, 2),
        "action_type_accuracy": round(action_acc, 2), "action_type_f1": round(action_f1, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "intent_latency_ms": round(avg_intent_latency, 2),
        "category_latency_ms": round(avg_cat_latency, 2)
    }

def main():
    if os.path.exists("/workspace"):
        dataset_path = Path("/workspace/text_nlu/datasets/nlu_benchmark.json")
    else:
        dataset_path = ROOT / "text_nlu" / "datasets" / "nlu_benchmark.json"
        
    # Read samples
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
                
    print(f"✅ Loaded {len(samples)} benchmark samples from {dataset_path}.")
    
    results = {}
    results["tfidf"] = evaluate_backend("TF-IDF", samples, backend_override="tfidf")
    results["phobert"] = evaluate_backend("PhoBERT", samples, backend_override="encoder")
    results["qwen_base"] = evaluate_backend("Qwen 2.5 Base", samples, backend_override="qwen_base")
    results["qwen_lora"] = evaluate_backend("Qwen 2.5 LoRA", samples, backend_override="qwen_lora")
        
    # Save output securely
    if os.path.exists("/storage"):
        output_file = Path("/storage/nlu_models/nlu_benchmark_results.json")
    else:
        output_file = ROOT / "text_nlu" / "datasets" / "nlu_benchmark_results.json"
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 Benchmark successfully completed! Results written to: {output_file}")
    
    print("\n" + "="*160)
    print("                                              BẢNG SO SÁNH HIỆU NĂNG NLU ĐÁNH GIÁ LUẬN VĂN")
    print("="*160)
    print(f"{'Mô hình NLU':<15} | {'Intent (Acc/F1)':<18} | {'Action (Acc/F1)':<18} | {'Category (Acc/F1)':<18} | {'Record (Acc/F1)':<18} | {'Stage 1 (ms)':<13} | {'Stage 2 (ms)':<13} | {'Total (ms)':<10}")
    print("-" * 160)
    for model_key, res in results.items():
        name = "TF-IDF" if model_key == "tfidf" else "PhoBERT" if model_key == "phobert" else ("Qwen Base" if model_key == "qwen_base" else "Qwen LoRA")
        intent_str = f"{res['intent_accuracy']:.1f}%/{res['intent_f1']:.1f}%"
        action_str = f"{res['action_type_accuracy']:.1f}%/{res['action_type_f1']:.1f}%"
        cat_str = f"{res['category_accuracy']:.1f}%/{res['category_f1']:.1f}%"
        rec_str = f"{res['record_type_accuracy']:.1f}%/{res['record_type_f1']:.1f}%"
        
        s1 = f"{res['intent_latency_ms']:.1f}"
        s2 = f"{res['category_latency_ms']:.1f}"
        total = f"{res['avg_latency_ms']:.1f}"
        
        print(f"{name:<15} | {intent_str:<18} | {action_str:<18} | {cat_str:<18} | {rec_str:<18} | {s1:<13} | {s2:<13} | {total:<10}")
    print("="*160)

if __name__ == "__main__":
    main()
