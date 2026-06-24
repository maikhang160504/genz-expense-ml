import pandas as pd
from pathlib import Path

csv_path = Path("d:/Luan-Van/Project/intent_action_10000(1).csv")
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))
print("Unique intents:", df['intent'].value_counts().to_dict())
print("Unique action_types:")
for k, v in df['action_type'].value_counts().items():
    print(f"  {k}: {v}")
