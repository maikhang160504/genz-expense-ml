import shutil
import pandas as pd
import json
import re
import os

# Paths
datasets_dir = r"d:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets"
intent_action_path = os.path.join(datasets_dir, "intent_action.csv")
intent_action_bak = os.path.join(datasets_dir, "intent_action.csv.bak")
ner_dataset_path = os.path.join(datasets_dir, "ner_dataset.jsonl")
ner_dataset_bak = os.path.join(datasets_dir, "ner_dataset.jsonl.bak")

print("Starting dataset cleanup...")

# 1. Backup intent_action.csv
if not os.path.exists(intent_action_bak):
    shutil.copyfile(intent_action_path, intent_action_bak)
    print(f"Backed up intent_action.csv to {intent_action_bak}")

# 2. Clean intent_action.csv
df = pd.read_csv(intent_action_path)
initial_len = len(df)

# Drop EXPORT_DATA
df = df[df["action_type"] != "EXPORT_DATA"]
after_drop_len = len(df)
print(f"Dropped {initial_len - after_drop_len} rows of EXPORT_DATA from intent_action.csv")

def normalize_action_type(row):
    act = row["action_type"]
    text_lower = str(row["text"]).lower()
    
    if act in ["Report", "REPORT_GENERAL", "REPORT_COMPARE"]:
        return "REPORT_GENERAL"
    elif act in ["Search", "SEARCH_RECORD"]:
        return "SEARCH_RECORD"
    elif act == "Setting":
        return "SYSTEM_SETTING"
    elif act == "Edit":
        delete_keywords = ["xóa", "xoa", "bỏ", "bo", "hủy", "huy", "gỡ", "go"]
        if any(kw in text_lower for kw in delete_keywords):
            return "DELETE_RECORD"
        else:
            return "UPDATE_RECORD"
    return act

df["action_type"] = df.apply(normalize_action_type, axis=1)

# Clean polluted REPORT_GENERAL rows containing money values (they should be in intent_record instead)
_MONEY_PATTERN = re.compile(
    r"\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|tr|triệu|trieu|củ|cu)\b",
    re.I,
)
before_money_drop = len(df)
df = df[~((df["action_type"] == "REPORT_GENERAL") & df["text"].astype(str).map(lambda t: bool(_MONEY_PATTERN.search(t))))]
print(f"Dropped {before_money_drop - len(df)} polluted REPORT_GENERAL rows containing money from intent_action.csv")

df.to_csv(intent_action_path, index=False)
print("Normalized intent_action.csv action types. New distribution:")
print(df["action_type"].value_counts())

# 3. Backup ner_dataset.jsonl
if not os.path.exists(ner_dataset_bak):
    shutil.copyfile(ner_dataset_path, ner_dataset_bak)
    print(f"Backed up ner_dataset.jsonl to {ner_dataset_bak}")

# 4. Clean ner_dataset.jsonl
export_pattern = re.compile(r"(?:xuất\b(?! đề)|export\b)", re.IGNORECASE)
cleaned_lines = []
removed_count = 0

with open(ner_dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if export_pattern.search(item["text"]):
            removed_count += 1
        else:
            cleaned_lines.append(line)

with open(ner_dataset_path, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print(f"Removed {removed_count} export-related lines from ner_dataset.jsonl")
print(f"Saved cleaned ner_dataset.jsonl with {len(cleaned_lines)} lines")
print("Cleanup complete!")
