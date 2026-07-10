import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "intent_action.csv"

if csv_path.is_file():
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    initial_len = len(df)
    
    # Drop legacy actions
    legacy_actions = ["SET_INCOME", "UPDATE_RECORD", "DELETE_RECORD"]
    df = df[~df["action_type"].isin(legacy_actions)]
    
    final_len = len(df)
    removed = initial_len - final_len
    
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Cleaned intent_action.csv: removed {removed} rows of legacy actions. New total rows: {final_len}")
    print("New action type distribution:")
    print(df["action_type"].value_counts())
else:
    print("intent_action.csv not found")
