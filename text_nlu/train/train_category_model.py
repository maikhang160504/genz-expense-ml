import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.text_preprocessing import clean_category_text, vi_tokenize

DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
RECORD_CSV = DATA_DIR / "intent_record.csv"
MODEL_PATH = MODEL_DIR / "category_model.joblib"

# NOTE: If you later switch to a distance-based algorithm (e.g., KNN, SVM with RBF),
# consider using StandardScaler or similar normalization.


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    # Use only text and label (category)
    df = df[["text", "label"]].copy()
    df.dropna(subset=["text", "label"], inplace=True)
    return df


def train_model(df: pd.DataFrame) -> tuple[TfidfVectorizer, LinearSVC]:
    X = df["text"].astype(str).map(clean_category_text)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        tokenizer=vi_tokenize,
        ngram_range=(1, 3),
        max_features=8000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC(
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("classification_report")
    print(classification_report(y_test, y_pred))
    print("confusion_matrix")
    print(confusion_matrix(y_test, y_pred))

    # === Save metrics to JSON for thesis evidence ===
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    per_class = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    acc = float(accuracy_score(y_test, y_pred))

    metrics_out = MODEL_DIR / "tfidf_category_metrics.json"
    metrics_out.write_text(json.dumps({
        "task": "category",
        "accuracy": round(acc, 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf1), 4),
        "per_class_report": {k: v for k, v in per_class.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved TF-IDF category metrics -> {metrics_out}")

    return vectorizer, model


def save_model(vectorizer: TfidfVectorizer, model: LinearSVC) -> None:
    payload = {
        "vectorizer": vectorizer,
        "model": model,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved category model to {MODEL_PATH}")


def main() -> None:
    df = load_data()
    vectorizer, model = train_model(df)
    save_model(vectorizer, model)

    # Quick sanity check
    sample = "di ca phe 39k"
    sample_vec = vectorizer.transform([sample])
    print("sample:", sample, "->", model.predict(sample_vec)[0])


if __name__ == "__main__":
    main()
