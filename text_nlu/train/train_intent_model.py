import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.text_preprocessing import vi_tokenize

# NOTE: If you later switch to a distance-based algorithm (e.g., KNN, SVM with RBF),
# consider using StandardScaler or similar normalization.

DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
RECORD_CSV = DATA_DIR / "intent_record.csv"
ACTION_CSV = DATA_DIR / "intent_action.csv"
CHITCHAT_CSV = DATA_DIR / "intent_chitchat.csv"
MODEL_PATH = MODEL_DIR / "intent_model.joblib"

MONEY_RE = re.compile(
    r"(\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|triệu|trieu|củ|cu))",
    re.IGNORECASE,
)


def load_data() -> pd.DataFrame:
    # Record: create intent column as Record
    df_record = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    df_record = df_record[["text"]].copy()
    df_record["intent"] = "Record"

    # Action: keep intent column as Action
    df_action = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    df_action = df_action[["text", "intent"]].copy()

    # Chitchat: keep intent column as Chitchat
    df_chat = pd.read_csv(CHITCHAT_CSV, encoding="utf-8-sig")
    df_chat = df_chat[["text", "intent"]].copy()

    # Oversample Chitchat (lớp nhỏ) để giảm lệch; LogisticRegression vẫn dùng class_weight balanced.
    n_chat = len(df_chat)
    n_action = len(df_action)
    target_chat = min(n_action, max(n_chat * 4, 6000))
    if n_chat < target_chat:
        extra = df_chat.sample(n=target_chat - n_chat, replace=True, random_state=42)
        df_chat = pd.concat([df_chat, extra], ignore_index=True)

    df = pd.concat([df_record, df_action, df_chat], ignore_index=True)
    df.dropna(subset=["text", "intent"], inplace=True)
    return df


def train_model(df: pd.DataFrame) -> tuple[TfidfVectorizer, LogisticRegression, pd.DataFrame]:
    X = df["text"].astype(str)
    y = df["intent"].astype(str)

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
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("classification_report")
    print(classification_report(y_test, y_pred))
    print("confusion_matrix")
    print(confusion_matrix(y_test, y_pred))

    return vectorizer, model, df


def predict_intent(input_text: str, vectorizer: TfidfVectorizer, model: LogisticRegression) -> str:
    text = str(input_text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


def save_model(vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    payload = {
        "vectorizer": vectorizer,
        "model": model,
        "labels": ["Record", "Action", "Chitchat"],
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def main() -> None:
    df = load_data()
    print("Intent class counts (after Chitchat oversample):")
    print(df["intent"].value_counts())
    vectorizer, model, _ = train_model(df)
    save_model(vectorizer, model)

    # Quick sanity check
    sample = "an sang 30k"
    print("sample:", sample, "->", predict_intent(sample, vectorizer, model))


if __name__ == "__main__":
    main()
