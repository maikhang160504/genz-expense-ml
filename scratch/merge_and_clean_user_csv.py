import pandas as pd
from pathlib import Path

user_csv_path = Path("d:/Luan-Van/Project/intent_action_10000(1).csv")
main_csv_path = Path("d:/Luan-Van/Project/expense-ocr-nlu/text_nlu/datasets/intent_action.csv")

# 1. Load user CSV
df_user = pd.read_csv(user_csv_path)

# 2. Map labels to unified 13 categories
MAPPING = {
    "REPORT_GENERAL": "REPORT_GENERAL",
    "REPORT": "REPORT_GENERAL",
    "REPORT_INCOME": "REPORT_GENERAL",
    "REPORT_COMPARE": "REPORT_GENERAL",
    "REPORT_SAVINGS": "REPORT_GENERAL",
    "SEARCH_RECORD": "SEARCH_RECORD",
    "SEARCH": "SEARCH_RECORD",
    "SYSTEM_SETTING": "SYSTEM_SETTING",
    "SETTING": "SYSTEM_SETTING",
    "DELETE": "DELETE_RECORD",
    "DELETE_LAST": "DELETE_RECORD",
    "EDIT": "UPDATE_RECORD",
    "UPDATE_RECORD": "UPDATE_RECORD",
    "SUGGEST_BUDGET": "SUGGEST_BUDGET",
    "SUGGEST": "SUGGEST_BUDGET",
    "LIMIT": "SET_LIMIT",
    "SET_LIMIT": "SET_LIMIT",
    "TONE": "SET_TONE",
    "SET_TONE": "SET_TONE",
    "GOAL": "SET_GOAL",
    "SET_GOAL": "SET_GOAL",
    "SET_ALERT": "SET_ALERT",
    "SET_INCOME": "SET_INCOME",
    "SET_USERNAME": "SET_USERNAME",
}

def clean_action_type(val):
    if not isinstance(val, str):
        return val
    clean_val = val.strip().upper()
    if clean_val not in MAPPING:
        print(f"Warning: Unexpected label '{val}' mapped to itself")
    return MAPPING.get(clean_val, val)

df_user['action_type'] = df_user['action_type'].map(clean_action_type)
df_user['intent'] = 'Action'  # Normalize intent to "Action"

# 3. Load main CSV
df_main = pd.read_csv(main_csv_path)
print("Before merge: main CSV size =", len(df_main))

# 4. Concatenate
df_merged = pd.concat([df_main, df_user], ignore_index=True)

# 5. Deduplicate by 'text'
df_merged = df_merged.drop_duplicates(subset=["text"], keep="first")
print("After merge & deduplicate: main CSV size =", len(df_merged))

# 6. Save back to the main file
df_merged.to_csv(main_csv_path, index=False, encoding="utf-8-sig")
print("Successfully saved merged dataset to:", main_csv_path)
