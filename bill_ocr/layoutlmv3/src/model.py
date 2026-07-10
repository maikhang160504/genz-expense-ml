"""Model utilities for LayoutLMv3 pipeline.

- get_processor: loads the LayoutLMv3 processor.
- get_model: loads the LayoutLMv3 model for token classification.
- build_label_maps: builds label2id / id2label dictionaries from the dataset.
- preprocess_examples: tokenizes a batch and aligns label ids.
"""
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForTokenClassification


def get_processor(max_seq_length: int = 512):
    """Return a LayoutLMv3 processor with the given ``max_seq_length``."""
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    processor.tokenizer.model_max_length = max_seq_length
    return processor


def get_model(num_labels: int, id2label: dict, label2id: dict):
    """Instantiate a LayoutLMv3 model for token classification.

    Args:
        num_labels: number of distinct entity labels.
        id2label: mapping id -> label string.
        label2id: mapping label string -> id.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def build_label_maps(dataset) -> tuple[dict, dict]:
    """Create ``label2id`` and ``id2label`` from all labels in the dataset.

    The dataset is expected to have a ``labels`` column where each entry is a list
    of IOB tag strings.
    """
    label_set = set()
    for split in dataset:
        for example in dataset[split]:
            label_set.update(example["labels"])
    label_list = sorted(label_set)
    label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return label2id, id2label


def preprocess_examples(examples, processor, label2id, max_seq_length: int = 512):
    """Tokenize a batch of examples and align label ids.

    ``examples`` is a dict of lists as provided by the ``datasets`` library.
    The function returns a ``BatchEncoding`` with an added ``labels`` field.
    """
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in examples["image"]]
    words = examples["words"]
    boxes = examples["boxes"]
    labels = examples["labels"]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # Convert string labels to ids and pad to max_length with -100
    label_ids = [[label2id[l] for l in seq] for seq in labels]
    padded_labels = []
    seq_len = encoding.input_ids.shape[1]
    for seq in label_ids:
        padded = seq + [-100] * (seq_len - len(seq))
        padded_labels.append(padded)
    encoding["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return encoding
