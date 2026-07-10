"""
Phase 1.4: Create unified fine-tune dataset from 3 cleaned datasets.
Format: JSONL with Alpaca-style structure.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import json
from pathlib import Path

DS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DS_DIR / "unified_finetune.jsonl"

SYSTEM_INSTRUCTION = (
    "Ban la tro ly tai chinh Mimo. "
    "Phan tich cau noi cua nguoi dung va tra ve JSON voi cac truong: "
    "intent (Record/Action/Chitchat), va cac slot tuong ung. "
    "Neu la Record: tra ve label (danh muc), type (expense/income), is_money (true/false). "
    "Neu la Action: tra ve action_type, va cac slot nhu verb, category_code, value, goal_name, "
    "enabled, theme, verbal_style, time_range, query, note. "
    "Neu la Chitchat: tra ve sentiment (Positive/Negative/Neutral)."
)

def main():
    total = 0
    
    # ── 1. Action dataset ──
    print("Processing intent_action.csv...")
    df_action = pd.read_csv(DS_DIR / "intent_action.csv", low_memory=False)
    
    action_slot_cols = [
        "action_type", "verb", "category_code", "value", "goal_name",
        "enabled", "theme", "verbal_style", "time_range", "query", "note"
    ]
    
    action_entries = []
    for _, row in df_action.iterrows():
        output_dict = {"intent": "Action"}
        for col in action_slot_cols:
            if col in row and pd.notna(row[col]):
                output_dict[col] = str(row[col])
        
        entry = {
            "instruction": SYSTEM_INSTRUCTION,
            "input": str(row["text"]),
            "output": json.dumps(output_dict, ensure_ascii=False)
        }
        action_entries.append(entry)
    
    print(f"  Action: {len(df_action)} entries processed")

    # ── 2. Chitchat dataset ──
    print("Processing intent_chitchat.csv...")
    df_cc = pd.read_csv(DS_DIR / "intent_chitchat.csv", low_memory=False)
    
    chitchat_entries = []
    for _, row in df_cc.iterrows():
        output_dict = {"intent": "Chitchat"}
        if "sentiment" in row and pd.notna(row["sentiment"]):
            output_dict["sentiment"] = str(row["sentiment"])
        
        entry = {
            "instruction": SYSTEM_INSTRUCTION,
            "input": str(row["text"]),
            "output": json.dumps(output_dict, ensure_ascii=False)
        }
        chitchat_entries.append(entry)
    
    print(f"  Chitchat: {len(df_cc)} entries processed")

    # ── 3. Record dataset ──
    print("Processing intent_record.csv...")
    df_rec = pd.read_csv(DS_DIR / "intent_record.csv", low_memory=False)
    
    record_entries = []
    for _, row in df_rec.iterrows():
        output_dict = {"intent": "Record"}
        for col in ["label", "type", "is_money"]:
            if col in row and pd.notna(row[col]):
                val = row[col]
                if col == "is_money":
                    val = bool(val) if isinstance(val, (int, float)) else str(val)
                output_dict[col] = val if not isinstance(val, str) else str(val)
        
        entry = {
            "instruction": SYSTEM_INSTRUCTION,
            "input": str(row["text"]),
            "output": json.dumps(output_dict, ensure_ascii=False)
        }
        record_entries.append(entry)
    
    print(f"  Record: {len(df_rec)} entries processed")

    # Write all entries to output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for entry in action_entries + chitchat_entries + record_entries:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1
            
    print(f"\n{'=' * 60}")
    print(f"UNIFIED DATASET CREATED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"  Total records compiled: {total}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
