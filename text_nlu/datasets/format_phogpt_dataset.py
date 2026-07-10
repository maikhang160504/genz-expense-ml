# Save as text_nlu/datasets/format_phogpt_dataset.py
import json
import random
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATA_DIR / "phogpt_finetune.jsonl"
PROMPTS_JSON = DATA_DIR.parents[1] / "src" / "prompts" / "prompts.json"

# Set seed for reproducible dataset outputs
random.seed(42)

# Load prompts.json
if PROMPTS_JSON.is_file():
    with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
else:
    prompts_data = {}

SYS_PROMPT = (
    "Bạn là Mimo, trợ lý tài chính cá nhân thân thiện và thông thái của hệ thống spending-diary. "
    "Hãy phân tích ý định người dùng và trả về một cấu trúc JSON hợp lệ có dạng: "
    '{"intent": "...", "action_type": "...", "slots": {...}, "emotion": "...", "response": "..."}.'
)

def determine_emotion(intent, category, amount):
    if intent == "Chitchat":
        return "Happy"
    if intent == "Record":
        if amount and amount > 5000000:
            return "Sad"
        if category == "Entertainment" or category == "Shopping":
            return "Happy"
        return "Chill"
    if intent == "Action":
        if category and ("LIMIT" in str(category) or "ALERT" in str(category)):
            return "Alert"
        return "Chill"
    return "Chill"

def generate_mock_context_meta(intent, category, amount):
    time_of_day = random.choice(["Sáng sớm", "Trưa nắng", "Chiều muộn", "Đêm khuya"])
    weather = random.choice(["nắng ấm", "mưa rơi", "âm u"])
    if amount and amount > 2000000:
        wallet_health = "viêm màng túi"
    else:
        wallet_health = random.choice(["rủng rỉnh", "tạm ổn", "đang báo động"])
        
    days_to_payday = random.randint(1, 28)
    
    return {
        "time_of_day": time_of_day,
        "weather": weather,
        "wallet_health": wallet_health,
        "days_to_payday": f"{days_to_payday} ngày nữa tới kỳ lương"
    }

def generate_genz_response(text, intent, action_type, slots, emotion, context_meta):
    amount = slots.get("amount") or slots.get("value")
    category = slots.get("category") or slots.get("category_code")
    
    # Format amount representation (e.g., 50000 -> 50k)
    amount_str = ""
    if amount:
        try:
            val = float(amount)
            if val >= 1000000:
                amount_str = f"{val/1000000:.0f}tr" if val % 1000000 == 0 else f"{val/1000000:.1f}tr"
            else:
                amount_str = f"{val/1000:.0f}k" if val % 1000 == 0 else f"{val/1000:.1f}k"
        except:
            amount_str = str(amount)

    happy_slangs = prompts_data.get("mood_personas", {}).get("VUI_VE", {}).get("slang", ["vibe cực", "mãi đỉnh"])
    dan_doi_slangs = prompts_data.get("mood_personas", {}).get("DAN_DOI", {}).get("slang", ["ét ô ét", "nhức nhức cái đầu"])
    
    if emotion == "Happy":
        slang = random.choice(happy_slangs)
    elif emotion in ["Sad", "Alert"]:
        slang = random.choice(dan_doi_slangs)
    else:
        slang = "nha"

    time_vn = context_meta["time_of_day"]
    wallet_vn = context_meta["wallet_health"]

    if intent == "Record":
        is_income = any(x in text.lower() for x in ["thu", "nhận", "cộng", "lương", "salary", "bonus", "thưởng"])
        if is_income:
            return f"Ting ting! {time_vn} nhận ngay {amount_str or ''} rủng rỉnh nha bồ. {slang.capitalize()}! 💸"
        else:
            cat_vn = category
            if category == "Food": cat_vn = "ăn uống"
            elif category == "Shopping": cat_vn = "mua sắm"
            elif category == "Entertainment": cat_vn = "giải trí"
            elif category == "Transport": cat_vn = "đi lại"
            
            if emotion == "Happy":
                return f"Ghi nhận chi tiêu {cat_vn or ''} hết {amount_str or ''} {time_vn} nhé bồ. Ví đang {wallet_vn} mà chốt đơn xịn xò, {slang}! ✨"
            elif emotion == "Sad":
                return f"Ét ô ét! {time_vn} mà lại tốn thêm {amount_str or ''} cho {cat_vn or 'khoản này'} rồi, ví đang {wallet_vn} đó. {slang.capitalize()} luôn á! 😭"
            else:
                return f"Đã ghi nhận bồ chi {amount_str or ''} cho {cat_vn or 'giao dịch này'} rồi nhé bồ."

    elif intent == "Action":
        if action_type in ["REPORT_GENERAL", "REPORT_COMPARE"]:
            return f"Dạ bồ, {time_vn} để Mimo xuất báo cáo chi tiêu gửi bồ liền đây nè. {slang.capitalize()}! 📊"
        elif action_type in ["SET_LIMIT", "SET_GOAL", "ADD_GOAL"]:
            return f"Thiết lập hạn mức/mục tiêu tài chính mới thành công {time_vn} rồi nha. {slang.capitalize()}! Cố gắng giữ kỷ luật nhé bồ tèo. 💪"
        elif action_type == "SUGGEST_BUDGET":
            return f"Để Mimo xem tình hình ví bồ đang {wallet_vn} rồi gợi ý ngân sách chi tiêu hợp lý cho bồ nha. {slang.capitalize()}! 🧠"
        elif action_type == "EXPORT_DATA":
            return f"Dạ bồ, tệp xuất dữ liệu chi tiêu (Excel/CSV) của bồ đã sẵn sàng rồi nè! Tải ngay ở nút bên dưới nha bồ. 📥"
        else:
            return f"Đã thực hiện cấu hình hệ thống theo lệnh của bồ rồi nha."

    else:  # Chitchat
        chitchats = [
            f"Chào bạn ngoan xinh yêu của Mimo! {time_vn} thế nào bồ ơi, có gì vui kể Mimo nghe với! {slang.capitalize()}! 🥰",
            f"Hế lô bồ tèo! {time_vn} ví đang {wallet_vn}, bồ có cần Mimo ghi chép giúp chi tiêu gì không nè? {slang.capitalize()}! 💖",
            f"Mimo nghe đây bồ ơi! {time_vn} trời đẹp vibe cực, tám chuyện tài chính hay tâm sự mỏng gì không bồ? 💬",
            f"Hello bồ! Mimo luôn sẵn sàng hỗ trợ bồ quản lý ví tiền nhé! {slang.capitalize()}! 🚀"
        ]
        idx = (len(text) + random.randint(0, 10)) % len(chitchats)
        return chitchats[idx]

def format_row(text, intent, action_type=None, slots=None):
    slots = slots or {}
    action_type_val = action_type if pd.notna(action_type) else None
    
    amount = slots.get("amount") or slots.get("value")
    category = slots.get("category") or slots.get("category_code")
    emotion = determine_emotion(intent, category, amount)
    
    # Generate mock context metadata
    context_meta = generate_mock_context_meta(intent, category, amount)
    
    # Generate authentic Gen Z response utilizing the metadata
    response_text = generate_genz_response(text, intent, action_type_val, slots, emotion, context_meta)
            
    response_payload = {
        "intent": intent,
        "action_type": action_type_val,
        "slots": slots,
        "emotion": emotion,
        "response": response_text
    }
    
    context_meta_str = json.dumps(context_meta, ensure_ascii=False)
    prompt = f"### Câu hỏi:\n<<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\nNgữ cảnh hệ thống (CONTEXT_META): {context_meta_str}\nCâu thoại của người dùng: {text}\n\n### Trả lời:"
    response = json.dumps(response_payload, ensure_ascii=False, indent=2)
    return {"prompt": prompt, "response": response}

def main():
    samples = []
    seen_texts = set()
    
    # Read record CSV
    record_csv = DATA_DIR / "intent_record.csv"
    if record_csv.is_file():
        record_df = pd.read_csv(record_csv, encoding="utf-8-sig")
        # Deduplicate
        record_df = record_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        # Group and sample (cap at 500 per label)
        sampled_records = []
        for label, group in record_df.groupby("label"):
            sampled_group = group.sample(n=min(len(group), 500), random_state=42)
            sampled_records.append(sampled_group)
        if sampled_records:
            record_df = pd.concat(sampled_records).sample(frac=1, random_state=42)
            for _, row in record_df.iterrows():
                text_clean = str(row["text"]).strip()
                if text_clean.lower() not in seen_texts:
                    seen_texts.add(text_clean.lower())
                    slots = {"category": row.get("label"), "amount": row.get("amount")}
                    slots = {k: v for k, v in slots.items() if pd.notna(v)}
                    samples.append(format_row(row["text"], "Record", slots=slots))
        
    # Read action CSV
    action_csv = DATA_DIR / "intent_action.csv"
    if action_csv.is_file():
        action_df = pd.read_csv(action_csv, encoding="utf-8-sig")
        action_df = action_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        sampled_actions = []
        for a_type, group in action_df.groupby("action_type"):
            sampled_group = group.sample(n=min(len(group), 600), random_state=42)
            sampled_actions.append(sampled_group)
        if sampled_actions:
            action_df = pd.concat(sampled_actions).sample(frac=1, random_state=42)
            for _, row in action_df.iterrows():
                text_clean = str(row["text"]).strip()
                if text_clean.lower() not in seen_texts:
                    seen_texts.add(text_clean.lower())
                    slots = {"category": row.get("category_code"), "verb": row.get("verb"), "amount": row.get("amount")}
                    slots = {k: v for k, v in slots.items() if pd.notna(v)}
                    samples.append(format_row(row["text"], "Action", action_type=row["action_type"], slots=slots))
        
    # Read chitchat CSV
    chitchat_csv = DATA_DIR / "intent_chitchat.csv"
    if chitchat_csv.is_file():
        chitchat_df = pd.read_csv(chitchat_csv, encoding="utf-8-sig")
        chitchat_df = chitchat_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        chitchat_df = chitchat_df.sample(n=min(len(chitchat_df), 2000), random_state=42)
        for _, row in chitchat_df.iterrows():
            text_clean = str(row["text"]).strip()
            if text_clean.lower() not in seen_texts:
                seen_texts.add(text_clean.lower())
                samples.append(format_row(row["text"], "Chitchat"))
        
    # Shuffle final compiled dataset for training variety
    random.shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"PhoGPT Dataset formatted and saved: {len(samples)} lines in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
