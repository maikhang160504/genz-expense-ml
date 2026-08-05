"""Golden set benchmark for NLU 2-Stage architecture (Intent Stage 1 & Category Stage 2).

Evaluates F1 Macro, Accuracy, and Latency across 3 backends: tfidf, pho_bert, and llm_v2.
"""
from __future__ import annotations

import time
from typing import Any

# Golden set 1: Intent Stage 1 (Record / Action / Chitchat)
GOLDEN_SET_INTENT = [
    # Record (Spending/Income)
    {"text": "ăn phở 45k", "expected": "Record"},
    {"text": "vừa mua cái áo mới hết 300 ngàn", "expected": "Record"},
    {"text": "đổ xăng 50k", "expected": "Record"},
    {"text": "tiền nhà tháng này 3 triệu", "expected": "Record"},
    {"text": "lương tháng 4 về 15 triệu", "expected": "Record"},
    {"text": "mua thuốc cảm 120k", "expected": "Record"},
    {"text": "xem phim rap 180k", "expected": "Record"},
    {"text": "đóng học phí cho con 4 triệu", "expected": "Record"},
    {"text": "được thưởng dự án 2 triệu", "expected": "Record"},
    {"text": "siêu thị Vinmart 450k", "expected": "Record"},
    # Action (System/Report commands)
    {"text": "tháng này tôi tiêu hết bao nhiêu rồi?", "expected": "Action"},
    {"text": "đặt hạn mức ăn uống tháng này 3 triệu", "expected": "Action"},
    {"text": "tạo mục tiêu tiết kiệm 10 triệu mua xe trong 6 tháng", "expected": "Action"},
    {"text": "đổi giọng điệu sang nghiêm túc đi Mimo", "expected": "Action"},
    {"text": "bật cảnh báo chi tiêu cho tôi", "expected": "Action"},
    {"text": "tháng này so với tháng trước chi tiêu thay đổi thế nào?", "expected": "Action"},
    {"text": "gọi tôi là Khang nhé Mimo", "expected": "Action"},
    {"text": "liệt kê tất cả giao dịch tuần này", "expected": "Action"},
    {"text": "nạp thêm 500 nghìn vào quỹ tiết kiệm mua xe", "expected": "Action"},
    {"text": "chuyển giao diện sang chế độ tối", "expected": "Action"},
    # Chitchat (Social conversation)
    {"text": "hôm nay trời đẹp quá Mimo ơi", "expected": "Chitchat"},
    {"text": "xin chào trợ lý Mimo", "expected": "Chitchat"},
    {"text": "bạn có biết hát không?", "expected": "Chitchat"},
    {"text": "cảm ơn bạn nhiều nha", "expected": "Chitchat"},
    {"text": "tôi buồn quá", "expected": "Chitchat"},
]

# Golden set 2: Category Stage 2 (18 categories for Record intent)
GOLDEN_SET_CATEGORY = [
    {"text": "ăn sáng phở bò 45k", "expected": "Food"},
    {"text": "uống cà phê sữa đá 25k", "expected": "Food"},
    {"text": "đi taxi grab hết 60k", "expected": "Transport"},
    {"text": "đổ xăng xe máy 70k", "expected": "Transport"},
    {"text": "thuê phòng trọ tháng 5 hết 3.5tr", "expected": "Housing"},
    {"text": "trả tiền điện nước 500k", "expected": "Housing"},
    {"text": "mua áo khoác mùa đông 450k", "expected": "Shopping"},
    {"text": "mua sắm mỹ phẩm 300k", "expected": "Shopping"},
    {"text": "xem phim ở rạp CGV 180k", "expected": "Entertainment"},
    {"text": "đi hát karaoke cùng phòng 200k", "expected": "Entertainment"},
    {"text": "mua thuốc cảm sốt 80k", "expected": "Health"},
    {"text": "khám răng hết 500k", "expected": "Health"},
    {"text": "mua sách lập trình 150k", "expected": "Education"},
    {"text": "đóng học phí tiếng Anh 2 triệu", "expected": "Education"},
    {"text": "nhận lương tháng vừa rồi 15 triệu", "expected": "Salary"},
    {"text": "nhận tiền thưởng dự án 3 triệu", "expected": "Bonus"},
    {"text": "lãi suất ngân hàng được 200k", "expected": "Business"},
    {"text": "mua kem chống nắng 120k", "expected": "Beauty"},
]


def _calc_metrics(predictions: list[str], expected: list[str], total_time_sec: float) -> dict[str, Any]:
    """Calculate F1 Macro, Accuracy, and Latency ms."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(expected, predictions))
    f1 = float(f1_score(expected, predictions, average="macro", zero_division=0))
    avg_latency = int((total_time_sec / max(len(expected), 1)) * 1000)

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "latency_ms": avg_latency,
    }


def run_golden_set_benchmark() -> dict[str, Any]:
    """Run golden set benchmark for Stage 1 (Intent) and Stage 2 (Category) across 3 backends."""
    from src.nlu.models import get_inference_backend, _registry_inference_backend
    import os

    results: dict[str, dict[str, Any]] = {
        "stage1_intent": {},
        "stage2_category": {},
    }

    backends_to_test = ["tfidf", "pho_bert", "llm_v2"]

    for backend in backends_to_test:
        # 1. Evaluate Stage 1 (Intent)
        preds_intent: list[str] = []
        expect_intent: list[str] = [item["expected"] for item in GOLDEN_SET_INTENT]
        start_t1 = time.perf_counter()

        for item in GOLDEN_SET_INTENT:
            pred = _simulate_infer(item["text"], backend, target="intent")
            preds_intent.append(pred)

        time_t1 = time.perf_counter() - start_t1
        results["stage1_intent"][backend] = _calc_metrics(preds_intent, expect_intent, time_t1)

        # 2. Evaluate Stage 2 (Category)
        preds_cat: list[str] = []
        expect_cat: list[str] = [item["expected"] for item in GOLDEN_SET_CATEGORY]
        start_t2 = time.perf_counter()

        for item in GOLDEN_SET_CATEGORY:
            pred = _simulate_infer(item["text"], backend, target="category")
            preds_cat.append(pred)

        time_t2 = time.perf_counter() - start_t2
        results["stage2_category"][backend] = _calc_metrics(preds_cat, expect_cat, time_t2)

    return {
        "ok": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark": results,
    }


def _simulate_infer(text: str, backend: str, target: str) -> str:
    """Helper to predict intent or category using specified backend."""
    try:
        from src.nlu import pipeline
        from src.nlu.models import _get_path, _load_tfidf, _load_encoder
        from src.config import settings

        if backend == "llm_v2":
            from src.nlu.llm_intent_handler import run_llm_nlu_v2
            res = run_llm_nlu_v2(text)
            if target == "intent":
                return str(res.get("intent", "Unknown"))
            return str(res.get("category", "Others"))

        if target == "intent":
            if backend == "pho_bert":
                model_info = _load_encoder(_get_path(settings.INTENT_ENCODER_PATH))
            else:
                model_info = _load_tfidf(_get_path(settings.MODEL_PATH))
            intent, _, _ = pipeline.classify_intent(text, model_info)
            return str(intent)

        # target == "category"
        if backend == "pho_bert":
            cat_model = _load_encoder(_get_path(settings.CATEGORY_ENCODER_PATH))
        else:
            cat_model = _load_tfidf(_get_path(settings.CATEGORY_MODEL_PATH))
            
        if cat_model.get("backend") == "encoder" and cat_model.get("bundle"):
            from src.nlu.encoder_runtime import predict_category_encoder
            pred = predict_category_encoder(cat_model["bundle"], text)
        elif cat_model.get("backend") == "tfidf" and cat_model.get("vectorizer"):
            vec = cat_model["vectorizer"].transform([text])
            pred = cat_model["model"].predict(vec)[0]
        else:
            pred = "Others"
        return str(pred or "Others")

    except Exception as e:
        # Fallback default on error
        return "Record" if target == "intent" else "Others"
