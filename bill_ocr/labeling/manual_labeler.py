"""
Manual labeling script.
Allows the admin to review and modify auto-labeled results, and exports them to the verified dataset directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

EXPORTED_DIR = Path(__file__).resolve().parents[1] / "exported"


def load_auto_labeled_result(image_name: str) -> dict[str, Any] | None:
    """Load auto labeled results if they exist in output directory."""
    # Placeholders/logs where intermediate auto-labeled files are stored
    log_path = EXPORTED_DIR / f"{image_name}_auto.json"
    if log_path.is_file():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_verified_label(image_name: str, label_data: dict[str, Any]) -> None:
    """Save finalized verified label to exported/ directory."""
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORTED_DIR / f"{image_name}_verified.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)
    print(f"[MANUAL-LABELER] Saved verified label for {image_name} to {out_path}")


def review_label_cli(image_name: str, auto_data: dict[str, Any]) -> dict[str, Any]:
    """Interactive CLI menu to verify or correct labels."""
    print("=" * 60)
    print(f"REVIEWING LABELS FOR IMAGE: {image_name}")
    print("=" * 60)
    print(f"Original Total Cost (PICK): {auto_data.get('total_cost_original')}")
    print(f"Corrected Total Cost (LLM): {auto_data.get('total_cost_corrected')}")
    print(f"Assigned Category: {auto_data.get('bill_category')}")
    print("Items Extracted:")
    for idx, item in enumerate(auto_data.get("items", [])):
        print(f"  [{idx}] Raw: {item.get('raw')} -> Label: {item.get('label')} | Price: {item.get('price')}")
    
    verified_data = dict(auto_data)
    
    # CLI interactive loop (in production, called from endpoint or terminal script)
    # Since it is a CLI script, we provide prompt options
    while True:
        print("\nOptions:")
        print("1. Accept all auto-labels as is")
        print("2. Correct Category")
        print("3. Correct Total Cost")
        print("4. Edit Items")
        print("5. Exit without saving")
        
        choice = input("Enter choice (1-5): ").strip()
        if choice == "1":
            break
        elif choice == "2":
            new_cat = input(f"Enter new category (current: {verified_data.get('bill_category')}): ").strip()
            if new_cat:
                verified_data["bill_category"] = new_cat
        elif choice == "3":
            new_total = input(f"Enter new total cost: ").strip()
            if new_total.isdigit():
                verified_data["total_cost_corrected"] = int(new_total)
                verified_data["total_cost_fixed"] = True
        elif choice == "4":
            print("Editing items:")
            for idx, item in enumerate(verified_data.get("items", [])):
                print(f"[{idx}] {item.get('label')}: {item.get('price')}")
            item_idx = input("Enter item index to edit (or 'a' to add new, 'd' to delete, 'enter' to skip): ").strip()
            if item_idx.isdigit():
                i = int(item_idx)
                if 0 <= i < len(verified_data["items"]):
                    new_label = input(f"New label (current: {verified_data['items'][i]['label']}): ").strip()
                    new_price = input(f"New price (current: {verified_data['items'][i]['price']}): ").strip()
                    if new_label:
                        verified_data["items"][i]["label"] = new_label
                    if new_price.isdigit():
                        verified_data["items"][i]["price"] = int(new_price)
            elif item_idx.lower() == "a":
                raw_name = input("Raw name: ").strip()
                lbl_name = input("Label name: ").strip()
                price_val = input("Price: ").strip()
                if lbl_name and price_val.isdigit():
                    verified_data.setdefault("items", []).append({
                        "raw": raw_name or lbl_name,
                        "label": lbl_name,
                        "price": int(price_val)
                    })
            elif item_idx.lower() == "d":
                del_idx = input("Enter index to delete: ").strip()
                if del_idx.isdigit():
                    di = int(del_idx)
                    if 0 <= di < len(verified_data["items"]):
                        verified_data["items"].pop(di)
        elif choice == "5":
            print("Exited without saving.")
            return auto_data
            
    save_verified_label(image_name, verified_data)
    return verified_data
