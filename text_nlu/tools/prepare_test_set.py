import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "text_nlu" / "datasets" / "nlu_benchmark.json"
OUTPUT_FILE = ROOT / "text_nlu" / "datasets" / "nlu_test_120.json"

def main():
    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Total samples found: {len(data)}")
    
    # Filter out anything that isn't a dict with 'text'
    valid_data = [d for d in data if isinstance(d, dict) and "text" in d]
    
    # Sample 120
    random.seed(42)  # For reproducibility
    sample_size = min(120, len(valid_data))
    sampled_data = random.sample(valid_data, sample_size)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {sample_size} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
