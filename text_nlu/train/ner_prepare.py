from __future__ import annotations

import json
import random
from pathlib import Path

import spacy
from spacy.util import filter_spans
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "ner_dataset.jsonl"
TRAIN_PATH = ROOT / "datasets" / "ner_train.spacy"
DEV_PATH = ROOT / "datasets" / "ner_dev.spacy"


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_docbin(samples: list[dict], nlp) -> tuple[DocBin, int]:
    doc_bin = DocBin()
    skipped = 0
    for sample in samples:
        text = sample["text"]
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in sample.get("label", []):
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped += 1
                continue
            ents.append(span)
        doc.ents = filter_spans(ents)
        doc_bin.add(doc)
    return doc_bin, skipped


def main() -> None:
    random.seed(50)
    nlp = spacy.blank("xx")
    samples = load_samples(DATA_PATH)
    random.shuffle(samples)

    split_at = int(len(samples) * 0.9)
    train_samples = samples[:split_at]
    dev_samples = samples[split_at:]

    train_bin, train_skipped = build_docbin(train_samples, nlp)
    dev_bin, dev_skipped = build_docbin(dev_samples, nlp)

    train_bin.to_disk(TRAIN_PATH)
    dev_bin.to_disk(DEV_PATH)

    print(f"train: {len(train_samples)} samples, skipped spans: {train_skipped}")
    print(f"dev: {len(dev_samples)} samples, skipped spans: {dev_skipped}")
    print(f"train_path: {TRAIN_PATH}")
    print(f"dev_path: {DEV_PATH}")


if __name__ == "__main__":
    main()
