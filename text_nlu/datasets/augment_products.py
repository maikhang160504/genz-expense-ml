"""
Augment intent_record.csv with 10k product records using vietnamese_products_80000_simple.csv.
Requirements:
1. 2000 short consumer product names (1 to 2 words) + price.
2. 4000 general food/essentials products + prefixes/suffixes/companions.
3. 4000 product names + context words ("đi", "bạn bè", "người yêu", "tụ tập", "cf") -> Entertainment.
   Also includes subscription services (Netflix, Spotify) and rice (gạo) mapped to correct categories.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRODUCT_CSV = Path("d:/Luan-Van/Project/vietnamese_products_80000_simple.csv")
RECORD_CSV = ROOT / "intent_record.csv"


def clean_product_name(p: str) -> str:
    p = p.lower().strip()
    p = re.sub(r"[^\w\s\+]", "", p)
    return p


def get_short_name(p: str) -> str:
    p = clean_product_name(p)
    words = p.split()
    if not words:
        return ""
    # Strip common descriptor words from the end of the first 2 words
    # e.g., if words[1] is "nguyên", "dinh", "ít", "không", "loại", etc.
    if len(words) > 1 and words[1] in {"nguyên", "dinh", "ít", "không", "loại", "truyền", "cao", "trẻ"}:
        return words[0]
    return " ".join(words[:2])


def main() -> None:
    if not PRODUCT_CSV.exists():
        print(f"Error: {PRODUCT_CSV} not found!")
        return

    print("Loading product dataset...")
    prod_df = pd.read_csv(PRODUCT_CSV, encoding="utf-8")
    
    # Map 'Đồ dùng trẻ em' -> 'Essentials'
    prod_df["category"] = prod_df["category"].replace({"Đồ dùng trẻ em": "Essentials"})

    # Filter by category
    food_products = prod_df[prod_df["category"] == "Food"]["product_name"].dropna().tolist()
    essentials_products = prod_df[prod_df["category"] == "Essentials"]["product_name"].dropna().tolist()

    # Clean food products to remove raw rice ("gạo") to avoid Food vs Essentials conflict
    food_products = [
        p for p in food_products 
        if not ("gạo" in p.lower() or "gao" in p.lower()) or ("bánh" in p.lower() or "cháo" in p.lower())
    ]

    prices = [
        "10k", "15k", "18k", "20k", "25k", "28k", "30k", "32k", "35k", "38k",
        "40k", "45k", "50k", "60k", "70k", "80k", "90k", "100k", "120k", "150k",
        "180k", "200k", "250k", "300k", "500k", "1tr", "1.5tr", "2tr"
    ]
    
    companions = [
        "ba mẹ", "mẹ", "bố", "cha mẹ", "ông bà", "anh trai", "chị gái", "em út",
        "con", "bé", "người yêu", "bồ", "ny", "crush", "bạn gái", "bạn trai",
        "bạn thân", "đồng nghiệp", "sếp", "Minh Anh", "Tuấn", "Lan", "Hùng"
    ]

    augmented_rows: list[dict] = []
    random.seed(42)

    # 1. Generate targeted subscription service rows (300 rows) -> Entertainment
    subscriptions = ["netflix", "spotify", "youtube premium", "icloud", "capcut pro", "chatgpt plus"]
    sub_templates = [
        "thanh toán {product} {price}",
        "thanh toán {product} tháng {price}",
        "gia hạn {product} {price}",
        "gia hạn {product} tháng {price}",
        "mua {product} {price}",
        "mua {product} tháng này {price}",
        "mua gói {product} {price}",
        "đăng ký {product} {price}"
    ]
    for _ in range(300):
        sub = random.choice(subscriptions)
        tpl = random.choice(sub_templates)
        price = random.choice(prices)
        augmented_rows.append({
            "text": tpl.format(product=sub, price=price),
            "label": "Entertainment",
            "type": "expense",
            "is_money": 1
        })

    # 2. Generate targeted raw rice rows (300 rows) -> Essentials
    rice_items = ["gạo", "gạo thơm", "gạo tẻ", "gạo nếp", "bao gạo", "túi gạo"]
    rice_templates = [
        "mua {product} {price}",
        "mua {product} ăn {price}",
        "tiền mua {product} {price}",
        "mua giùm {product} {price}",
        "thanh toán {product} {price}",
        "mua thêm {product} {price}",
        "{product} hết {price}",
        "mua {product} cho gia đình {price}"
    ]
    for _ in range(300):
        rice = random.choice(rice_items)
        tpl = random.choice(rice_templates)
        price = random.choice(prices)
        augmented_rows.append({
            "text": tpl.format(product=rice, price=price),
            "label": "Essentials",
            "type": "expense",
            "is_money": 1
        })

    # Clean and list
    all_food_cleaned = list(set(clean_product_name(p) for p in food_products))
    all_ess_cleaned = list(set(clean_product_name(p) for p in essentials_products))
    
    # Add rice to all_ess_cleaned
    all_ess_cleaned.extend(rice_items)

    # 3. 2000 short consumer product names (1 to 2 words) + price.
    short_food = list(set(get_short_name(p) for p in food_products))
    short_ess = list(set(get_short_name(p) for p in essentials_products))
    
    # Add rice to short essentials
    short_ess.extend(rice_items)
    
    # Remove empty names
    short_food = [p for p in short_food if p]
    short_ess = [p for p in short_ess if p]

    print(f"Short food count: {len(short_food)}, Short essentials count: {len(short_ess)}")

    # We need to fill the remaining 1400 short product rows
    sample_short_food = random.sample(short_food, min(700, len(short_food)))
    sample_short_ess = random.sample(short_ess, min(700, len(short_ess)))

    for p in sample_short_food:
        price = random.choice(prices)
        augmented_rows.append({
            "text": f"{p} {price}",
            "label": "Food",
            "type": "expense",
            "is_money": 1
        })

    for p in sample_short_ess:
        price = random.choice(prices)
        augmented_rows.append({
            "text": f"{p} {price}",
            "label": "Essentials",
            "type": "expense",
            "is_money": 1
        })

    # 4. General food/essentials products + prefixes/suffixes/companions -> total 5000 rows
    food_templates = [
        "mua {product} {price}",
        "ăn {product} {price}",
        "uống {product} {price}",
        "order {product} {price}",
        "thanh toán {product} {price}",
        "mua {product} cho {companion} {price}",
        "ăn {product} cùng {companion} {price}",
        "uống {product} cùng {companion} {price}"
    ]

    ess_templates = [
        "mua {product} {price}",
        "thanh toán {product} {price}",
        "mua {product} cho {companion} {price}",
        "mua giùm {companion} {product} {price}",
        "mua thêm {product} {price}",
        "chi tiền mua {product} {price}"
    ]

    # Sample products for general food/essentials
    gen_food_samples = random.sample(all_food_cleaned, min(2500, len(all_food_cleaned)))
    gen_ess_samples = random.sample(all_ess_cleaned, min(2500, len(all_ess_cleaned)))

    for p in gen_food_samples:
        price = random.choice(prices)
        tpl = random.choice(food_templates)
        text = tpl.format(product=p, companion=random.choice(companions), price=price)
        augmented_rows.append({
            "text": text,
            "label": "Food",
            "type": "expense",
            "is_money": 1
        })

    for p in gen_ess_samples:
        price = random.choice(prices)
        tpl = random.choice(ess_templates)
        text = tpl.format(product=p, companion=random.choice(companions), price=price)
        augmented_rows.append({
            "text": text,
            "label": "Essentials",
            "type": "expense",
            "is_money": 1
        })

    # 5. 3000 rows of product name + context words like "đi", "bạn bè", "người yêu" -> Entertainment
    ent_base_products = [
        p for p in all_food_cleaned 
        if any(k in p for k in ["trà sữa", "cà phê", "lẩu", "nướng", "bia", "rượu", "cafe", "cf", "nước ngọt", "kem", "bánh tráng", "ốc", "trà", "sinh tố", "nước ép", "nước mía"])
    ]
    # Add subscriptions to entertainment products
    ent_base_products.extend(subscriptions)
    
    if len(ent_base_products) < 500:
        ent_base_products = all_food_cleaned

    ent_samples = random.sample(ent_base_products, min(3000, len(ent_base_products)))
    while len(ent_samples) < 3000:
        ent_samples.extend(random.sample(ent_base_products, min(3000 - len(ent_samples), len(ent_base_products))))

    ent_templates = [
        "đi {product} {price}",
        "đi uống {product} {price}",
        "đi ăn {product} cùng bạn bè {price}",
        "đi {product} với người yêu {price}",
        "tụ tập {product} {price}",
        "tụ tập bạn bè đi uống {product} {price}",
        "hẹn hò {product} {price}",
        "đi {product} với bạn {price}",
        "tụ tập ăn {product} {price}",
        "đi nhậu {product} {price}",
        "đi {product} cuối tuần {price}",
        "đi {product} cùng {companion} {price}",
        "gia hạn {product} {price}",
        "thanh toán {product} {price}"
    ]

    for p in ent_samples[:3000]:
        price = random.choice(prices)
        tpl = random.choice(ent_templates)
        text = tpl.format(product=p, companion=random.choice(companions), price=price)
        augmented_rows.append({
            "text": text,
            "label": "Entertainment",
            "type": "expense",
            "is_money": 1
        })

    # Adjust to exactly 10,000 rows
    while len(augmented_rows) < 10000:
        p = random.choice(short_food + short_ess)
        price = random.choice(prices)
        label = "Food" if p in short_food else "Essentials"
        augmented_rows.append({
            "text": f"{p} {price}",
            "label": label,
            "type": "expense",
            "is_money": 1
        })
    augmented_rows = augmented_rows[:10000]

    # Append to existing intent_record.csv
    print(f"Reading existing {RECORD_CSV}...")
    orig_df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    print(f"Original shape: {orig_df.shape}")

    new_rows_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([orig_df, new_rows_df], ignore_index=True)
    print(f"Saving combined dataset to {RECORD_CSV} (new shape: {combined_df.shape})...")
    combined_df.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print("Done!")


if __name__ == "__main__":
    main()
