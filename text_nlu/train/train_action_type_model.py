import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.text_preprocessing import vi_tokenize

DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
ACTION_CSV = DATA_DIR / "intent_action.csv"
MODEL_PATH = MODEL_DIR / "action_type_model.joblib"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    df = df[["text", "action_type"]].copy()
    df.dropna(subset=["text", "action_type"], inplace=True)
    return df


def train_model(df: pd.DataFrame) -> tuple[TfidfVectorizer, LinearSVC]:
    X = df["text"].astype(str)
    y = df["action_type"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        tokenizer=vi_tokenize,
        ngram_range=(1, 2),
        max_features=6000,
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

    return vectorizer, model


def save_model(vectorizer: TfidfVectorizer, model: LinearSVC) -> None:
    payload = {
        "vectorizer": vectorizer,
        "model": model,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved action type model to {MODEL_PATH}")


def main() -> None:
    df = load_data()
    vectorizer, model = train_model(df)
    save_model(vectorizer, model)

    # Quick sanity check
    sample = "tang han muc an uong them 200k"
    sample_vec = vectorizer.transform([sample])
    print("sample:", sample, "->", model.predict(sample_vec)[0])


if __name__ == "__main__":
    main()
