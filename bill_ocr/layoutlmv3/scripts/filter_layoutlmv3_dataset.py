#!/usr/bin/env python3
import os
import csv
import shutil
import ast
import argparse
from pathlib import Path

# Keywords for rule-based correction
KEYWORDS_TIMESTAMP = ['ngày', 'thời gian', 'giờ']
KEYWORDS_TOTAL_COST = ['tổng tiền', 'cộng tiền hàng', 'tổng cộng', 'thanh toán', 'tại quầy']

def validate_amount(text):
    # Basic numeric amount validator like validate_TOTAL_COST_amount
    cleaned = text.replace('.', '').replace(',', '').replace(' ', '').replace('đ', '').replace('d', '')
    return cleaned.isdigit()

def filter_dataset(csv_file, img_dir, output_csv, output_img_dir):
    csv_file = Path(csv_file)
    img_dir = Path(img_dir)
    output_csv = Path(output_csv)
    output_img_dir = Path(output_img_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV from {csv_file}...")
    with open(csv_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"Total rows read: {len(rows)}")

    output_rows = [header]
    total_wrong_key = 0
    total_mismatched = 0
    total_ignored = 0
    total_missing_img = 0

    for idx, row in enumerate(rows):
        img_name = row[0]
        img_path = img_dir / img_name
        if not img_path.is_file():
            total_missing_img += 1
            continue

        try:
            polygons = ast.literal_eval(row[1])
        except Exception as e:
            print(f"Row {idx+1} ({img_name}): Failed to parse polygons: {e}")
            total_ignored += 1
            continue

        texts = str(row[2]).split('|||')
        labels = str(row[3]).split('|||')

        # Rule 1: Drop mismatched list lengths
        if len(polygons) != len(texts) or len(polygons) != len(labels):
            total_mismatched += 1
            continue

        # Rule 2: Clean up label typos & enforce correct schema mapping
        clean_labels = []
        clean_polygons = []
        for j, label in enumerate(labels):
            clean_label = str(label).strip()
            val_lower = str(texts[j]).lower()

            # Fix typos
            if clean_label == "TIMESTAMPS":
                clean_label = "TIMESTAMP"
                total_wrong_key += 1
            elif clean_label == "TOTAL_TOTAL_COST":
                clean_label = "TOTAL_COST"
                total_wrong_key += 1

            # Keyword rules (only correct if the original label is not SELLER or ADDRESS to protect them)
            if clean_label not in ['SELLER', 'ADDRESS']:
                for kw in KEYWORDS_TIMESTAMP:
                    if kw in val_lower and clean_label != 'TIMESTAMP':
                        clean_label = 'TIMESTAMP'
                        total_wrong_key += 1

                for kw in KEYWORDS_TOTAL_COST:
                    if kw in val_lower and clean_label != 'TOTAL_COST':
                        clean_label = 'TOTAL_COST'
                        total_wrong_key += 1

            # Validate total cost amount numbers
            if validate_amount(val_lower) and clean_label == 'TIMESTAMP':
                clean_label = 'TOTAL_COST'
                total_wrong_key += 1

            clean_labels.append(clean_label)
            
            # Map category_id corresponding to label
            # Typical mapping: SELLER=15, ADDRESS=16, TIMESTAMP=17, TOTAL_COST=18
            cat_map = {'SELLER': 15, 'ADDRESS': 16, 'TIMESTAMP': 17, 'TOTAL_COST': 18}
            poly = polygons[j]
            if clean_label in cat_map:
                poly['category_id'] = cat_map[clean_label]
            clean_polygons.append(poly)

        # Rule 3: Heuristics - Filter by key counts (e.g. at least 2 unique keys, max 8 TOTAL_COST)
        num_keys = {'SELLER': 0, 'ADDRESS': 0, 'TIMESTAMP': 0, 'TOTAL_COST': 0}
        for k in clean_labels:
            if k in num_keys:
                num_keys[k] += 1

        ignore_file = False
        if num_keys['TOTAL_COST'] > 8:
            ignore_file = True

        total_exist_keys = sum(1 for k, v in num_keys.items() if v > 0)
        if total_exist_keys < 2:
            ignore_file = True

        if ignore_file:
            total_ignored += 1
            continue

        # Save corrected values back to row
        row[1] = str(clean_polygons)
        row[2] = '|||'.join(texts)
        row[3] = '|||'.join(clean_labels)
        output_rows.append(row)

        # Copy image file to output directory
        shutil.copy2(img_path, output_img_dir / img_name)

    print("\n--- Preprocessing Results ---")
    print(f"Initial rows: {len(rows)}")
    print(f"Saved cleaned rows: {len(output_rows) - 1}")
    print(f"Mismatched rows dropped: {total_mismatched}")
    print(f"Rows ignored by rules/heuristics: {total_ignored}")
    print(f"Rows with missing images: {total_missing_img}")
    print(f"Total wrong labels fixed: {total_wrong_key}")

    # Write cleaned CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
    print(f"Success: Cleaned CSV saved to: {output_csv}")
    print(f"Success: Cleaned images saved to: {output_img_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter and clean MC_OCR training dataset before LayoutLMv3 training")
    parser.add_argument("--csv", type=str, required=True, help="Path to input raw training CSV")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to raw training images directory")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to output cleaned CSV file")
    parser.add_argument("--out_img_dir", type=str, required=True, help="Path to output cleaned images directory")
    args = parser.parse_args()

    filter_dataset(args.csv, args.img_dir, args.out_csv, args.out_img_dir)
