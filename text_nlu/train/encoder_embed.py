"""PhoBERT embedding helpers for encoder training (self-contained for Kaggle)."""
from __future__ import annotations

from typing import Any

import numpy as np

_HF_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_hf(model_name: str):
    if model_name not in _HF_CACHE:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name, use_safetensors=False)
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
