"""
Encoder (PhoBERT / XLM-R) + Logistic đã hiệu chỉnh (CalibratedClassifierCV).

Chỉ import torch/transformers khi gọi embed / predict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_HF_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_hf(model_name: str):
    if model_name not in _HF_CACHE:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _HF_CACHE[model_name] = (tok, model)
    return _HF_CACHE[model_name]


def embed_texts(
    texts: list[str],
    model_name: str,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    import torch

    tokenizer, model = _get_hf(model_name)
    device = next(model.parameters()).device
    outs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[:, 0, :]
        outs.append(hidden.cpu().numpy().astype(np.float32))
    return np.vstack(outs)


def embed_one(text: str, model_name: str, max_length: int = 128) -> np.ndarray:
    return embed_texts([text], model_name, max_length=max_length, batch_size=1)


def predict_intent_encoder(bundle: dict[str, Any], text: str, money_re) -> tuple[str, float | None, dict[str, float]]:
    """money_re: compiled regex từ pipeline (câu có tiền không được Chitchat)."""
    clf = bundle["classifier"]
    name = bundle["encoder_model_name"]
    x = embed_one(text, name)
    pred_raw = str(clf.predict(x)[0])
    dist: dict[str, float] = {}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)[0]
        for c, p in zip(clf.classes_, proba):
            dist[str(c)] = float(p)
    intent = pred_raw
    if money_re.search(text) and pred_raw == "Chitchat":
        intent = "Record"
    conf = dist.get(str(intent), max(dist.values())) if dist else None
    return intent, conf, dist


def predict_action_type_encoder(bundle: dict[str, Any], text: str) -> str:
    clf = bundle["classifier"]
    x = embed_one(text, bundle["encoder_model_name"])
    return str(clf.predict(x)[0])


def predict_sentiment_encoder(bundle: dict[str, Any], text: str) -> str:
    clf = bundle["classifier"]
    x = embed_one(text, bundle["encoder_model_name"])
    return str(clf.predict(x)[0])


def predict_record_type_encoder(bundle: dict[str, Any], text: str) -> str:
    """Trả 'income' hoặc 'expense' (lowercase)."""
    clf = bundle["classifier"]
    x = embed_one(text, bundle["encoder_model_name"], max_length=96)
    return str(clf.predict(x)[0]).lower()


def predict_category_encoder(bundle: dict[str, Any], text: str) -> str:
    clf = bundle["classifier"]
    x = embed_one(text, bundle["encoder_model_name"], max_length=96)
    return str(clf.predict(x)[0])


_PHOBERT_SENTIMENT_CACHE: dict[str, tuple[Any, Any, dict[str, int], int]] = {}


def predict_sentiment_phobert(model_dir: str, text: str) -> str:
    """Fine-tuned PhoBERT sequence classification (3 lớp sentiment)."""
    import json

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_dir not in _PHOBERT_SENTIMENT_CACHE:
        meta_path = Path(model_dir) / "nlu_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        max_len = int(meta.get("max_length", 96))
        label2id = meta.get("label2id") or {}
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, use_safetensors=True
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _PHOBERT_SENTIMENT_CACHE[model_dir] = (tok, model, label2id, max_len)

    tok, model, label2id, max_len = _PHOBERT_SENTIMENT_CACHE[model_dir]
    id2label = {int(v): k for k, v in label2id.items()} if label2id else {
        i: lab for i, lab in enumerate(getattr(model.config, "id2label", {}) or {})
    }
    device = next(model.parameters()).device
    enc = tok(
        text,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id = int(logits.argmax(dim=-1).item())
    if pred_id in id2label:
        return str(id2label[pred_id])
    return str(model.config.id2label.get(pred_id, "Neutral"))
