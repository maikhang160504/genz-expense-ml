"""
Synthetic dataset generator to expand NLU datasets to 20,000 samples.
Generates:
- intent_action.csv (20,000 unique rows of Action intents across 13 classes)
- ner_dataset.jsonl (20,000 unique rows with precise character offsets for spaCy NER training)
Supports Vietnamese dates, weekdays, Gen Z slang, abbreviations, and colloquial styles.
"""
from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTION_CSV_PATH = ROOT / "intent_action.csv"
NER_JSONL_PATH = ROOT / "ner_dataset.jsonl"

# 13 consolidated action type classes
ACTION_TYPES = [
    "REPORT_GENERAL",
    "SET_LIMIT",
    "SET_GOAL",
    "ADD_GOAL",
    "DELETE_RECORD",
    "SET_TONE",
    "SEARCH_RECORD",
    "SUGGEST_BUDGET",
    "SYSTEM_SETTING",
    "SET_USERNAME",
    "SET_INCOME",
    "UPDATE_RECORD",
    "SET_ALERT",
]

# Vocabulary Lists
PREFIXES = [
    "", "", "", "mimo ơi", "Mimo ơi", "bot ơi", "trợ lý ơi", "nè bot", "làm ơn",
    "ét ô ét", "gét gô", "vô tri", "mimo", "bot", "hey mimo", "cho hỏi", "giúp mình"
]

SUFFIXES = [
    "", "", "", "nhé", "nha", "nhe", "đi", "nha Mimo", "nhé Mimo", "giùm mình", "nha bạn",
    "vibe đỉnh", "vô tri", "cháy phố", "slay", "healing", "kiếp nạn thứ 82", "đỉnh nóc kịch trần",
    "hết nước chấm", "ầm ầm", "chill chill", "xịn xò"
]

CATEGORIES = [
    ("ăn uống", "Food"), ("ăn trưa", "Food"), ("ăn sáng", "Food"), ("cơm trưa", "Food"),
    ("phở bò", "Food"), ("cà phê", "Food"), ("cf", "Food"), ("cafe", "Food"),
    ("trà sữa", "Food"), ("bún bò", "Food"), ("đồ ăn", "Food"), ("mua đồ ăn", "Food"),
    ("đi lại", "Transport"), ("xăng xe", "Transport"), ("grab", "Transport"),
    ("bebike", "Transport"), ("xe ôm", "Transport"), ("vé xe buýt", "Transport"),
    ("đổ xăng", "Transport"), ("taxi", "Transport"), ("vé máy bay", "Transport"),
    ("mua sắm", "Shopping"), ("shopping", "Shopping"), ("shopee", "Shopping"),
    ("lazada", "Shopping"), ("quần áo", "Shopping"), ("giày sneaker", "Shopping"),
    ("mỹ phẩm", "Beauty"), ("son môi", "Beauty"), ("balo", "Shopping"),
    ("giải trí", "Entertainment"), ("chơi game", "Entertainment"), ("netflix", "Entertainment"),
    ("spotify", "Entertainment"), ("xem phim", "Entertainment"), ("cgv", "Entertainment"),
    ("tiền nhà", "Housing"), ("tiền điện", "Housing"), ("tiền nước", "Housing"),
    ("wifi", "Housing"), ("tiền mạng", "Housing"), ("thuê phòng trọ", "Housing"),
    ("dầu gội", "Essentials"), ("bột giặt", "Essentials"), ("bách hóa xanh", "Essentials"),
    ("y tế", "Health"), ("thuốc men", "Health"), ("khám bệnh", "Health"),
    ("cắt tóc", "Beauty"), ("làm nail", "Beauty"), ("học tập", "Education"),
    ("học phí", "Education"), ("mua sách", "Education"), ("khóa học", "Education"),
    ("quà cáp", "Social"), ("mừng cưới", "Social"), ("quà sinh nhật", "Social"),
    ("từ thiện", "Charity"), ("quyên góp", "Charity"), ("đầu tư", "Investment"),
    ("mua vàng", "Investment"), ("chứng khoán", "Investment"), ("coin", "Investment"),
    ("tiết kiệm", "Savings"), ("gửi tiết kiệm", "Savings"), ("lương", "Salary"),
    ("ting ting", "Salary"), ("bán hàng", "Business"), ("kinh doanh", "Business")
]

TIMES = [
    "hôm nay", "hom nay", "hn", "hôm qua", "hom qua", "hq", "hôm kia", "ngay mai", "ngày mai",
    "tuần này", "tuan nay", "tuần trước", "tuan truoc", "tháng này", "thang nay", "thg nay",
    "tháng trước", "thang truoc", "thg truoc", "tháng sau", "thang sau", "năm nay", "nam nay",
    "năm trước", "nam truoc", "ngày 22/06/2026", "22-06-2026", "22/06", "ngày 5 tháng 5",
    "thg 5", "tháng 12", "thứ 2", "thu 2", "t2", "thứ hai", "thứ 3", "thu 3", "t3", "thứ ba",
    "thứ 4", "t4", "thứ tư", "thứ 5", "t5", "thứ năm", "thứ 6", "t6", "thứ sáu", "thứ 7", "t7",
    "thứ bảy", "chủ nhật", "chu nhat", "cn", "cuối tuần", "cuoi tuan", "ngày thường", "ngày lễ",
    "dạo này", "mấy hôm nay", "gần đây", "hôm nọ"
]

AMOUNTS = [
    "10k", "15k", "20k", "25k", "30k", "35k", "40k", "50k", "60k", "70k", "80k", "90k", "100k",
    "120k", "150k", "180k", "200k", "250k", "300k", "400k", "500k", "600k", "700k", "800k", "900k",
    "1tr", "1.2tr", "1.5tr", "1.8tr", "2tr", "2.5tr", "3tr", "4tr", "5tr", "10tr", "15tr", "20tr",
    "100K", "200K", "500K", "50 cành", "100 cành", "200 cành", "500 cành", "1 củ", "2 củ", "5 củ", "10 củ",
    "50.000đ", "100.000 đồng", "200.000 vnd", "500.000 vnđ", "50000", "100000", "500000"
]

VERBS_SET = ["đặt", "cài", "thiết lập", "set", "cập nhật", "đặt lại", "đổi", "thay đổi", "cai", "doi"]
VERBS_ADD = ["tăng", "thêm", "cộng thêm", "nâng", "add", "bổ sung"]
VERBS_SUB = ["giảm", "bớt", "hạ", "sub", "cắt bớt"]

TARGETS_LIMIT = ["hạn mức", "giới hạn", "ngân sách", "quota", "han muc", "gioi han", "ngan sach"]
TARGETS_GOAL = ["mục tiêu", "kế hoạch tiết kiệm", "heo đất", "mục tiêu tiết kiệm", "muc tieu"]
TARGETS_ALERT = ["cảnh báo", "thông báo", "báo động", "cảnh báo hạn mức"]

TONES = [
    "dễ thương", "hài hước", "nghiêm túc", "châm chọc", "dặn dỗi", "funny", "gentle", "serious",
    "sarcastic", "strict", "đáng yêu", "giận dỗi", "đồng cảm", "lạnh lùng"
]

THEMES = [
    "chế độ tối", "giao diện tối", "dark mode", "giao diện ban đêm", "chế độ sáng", "giao diện sáng",
    "light mode", "giao diện ban ngày", "darkmode", "lightmode"
]

NAMES = [
    "An", "Khang", "Vy", "Linh", "Nam", "Dũng", "Trang", "Minh", "Huy", "Hoàng", "Tùng", "Thảo",
    "Hà", "Phương", "Phúc", "Kiệt", "Dương", "Ngọc", "Sơn", "Lâm", "Hải", "Khánh", "Mimo Master", "boss"
]

# Action indicator verbs for ACTION_TYPE labels in NER
INDICATORS_REPORT = ["báo cáo", "thống kê", "so sánh", "biểu đồ", "tổng chi", "báo cáo chi tiêu", "bao cao", "thong ke", "so sanh"]
INDICATORS_SEARCH = ["tìm", "kiếm", "tìm kiếm", "tra cứu", "lọc", "tìm các", "tim kiem", "tra cuu", "loc"]
INDICATORS_DELETE = ["xóa", "hủy", "bỏ", "xóa giao dịch", "hủy khoản", "xoa", "huy", "bo"]
INDICATORS_SETTING = ["mở", "vào", "bật", "tắt", "mở màn hình", "mo", "vao", "bat", "tat"]

def build_ner_sample(template: str, params: dict[str, tuple[str, str]]) -> dict:
    """
    Template parser that formats strings left-to-right and records precise 
    character offsets for entities.
    """
    pattern = re.compile(r"\{([A-Z_]+)\}")
    parts = []
    last_idx = 0
    for match in pattern.finditer(template):
        placeholder = match.group(1)
        parts.append(template[last_idx:match.start()])
        if placeholder in params:
            val_str, entity_name = params[placeholder]
            parts.append((val_str, entity_name))
        else:
            parts.append(match.group(0))
        last_idx = match.end()
    parts.append(template[last_idx:])
    
    final_text = ""
    label_spans = []
    for part in parts:
        if isinstance(part, tuple):
            val_str, entity_name = part
            start_idx = len(final_text)
            final_text += val_str
            end_idx = len(final_text)
            if entity_name:
                label_spans.append([start_idx, end_idx, entity_name])
        else:
            final_text += part
            
    # Clean redundant spaces
    cleaned_text = " ".join(final_text.split())
    # Re-calculate spans due to whitespace cleaning
    restored_spans = []
    for s_idx, e_idx, label in label_spans:
        prefix = final_text[:s_idx]
        cleaned_prefix = " ".join(prefix.split())
        start = len(cleaned_prefix)
        # handle starting space if prefix is not empty and ended with spaces
        if prefix and prefix[-1].isspace():
            start += 1
            
        span_text = final_text[s_idx:e_idx]
        cleaned_span_text = " ".join(span_text.split())
        end = start + len(cleaned_span_text)
        
        # Verify text matching
        if cleaned_text[start:end] == cleaned_span_text:
            restored_spans.append([start, end, label])
            
    return {"text": cleaned_text, "label": restored_spans}

def normalize_text_key(t: str) -> str:
    t = unicodedata.normalize("NFD", t or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(t.lower().split())

def main():
    print("=== NLU Dataset Large Generator ===")
    
    # 1. Load existing action samples to preserve hand-crafted validation rows
    existing_actions = []
    seen_action_texts = set()
    if ACTION_CSV_PATH.is_file():
        with open(ACTION_CSV_PATH, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt = row.get("text", "").strip()
                atype = row.get("action_type", "").strip()
                if txt and atype in ACTION_TYPES:
                    norm_k = normalize_text_key(txt)
                    if norm_k not in seen_action_texts:
                        seen_action_texts.add(norm_k)
                        existing_actions.append({"text": txt, "intent": "Action", "action_type": atype})
        print(f"Loaded {len(existing_actions)} clean action samples from {ACTION_CSV_PATH.name}")
    
    # 2. Load existing NER samples
    existing_ner = []
    seen_ner_texts = set()
    if NER_JSONL_PATH.is_file():
        with open(NER_JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    txt = data.get("text", "").strip()
                    if txt:
                        norm_k = normalize_text_key(txt)
                        if norm_k not in seen_ner_texts:
                            seen_ner_texts.add(norm_k)
                            existing_ner.append(data)
        print(f"Loaded {len(existing_ner)} clean NER samples from {NER_JSONL_PATH.name}")

    # Define generators for 13 action types
    generators = {}
    
    # REPORT_GENERAL
    generators["REPORT_GENERAL"] = [
        "{PREFIX} {ACTION_TYPE} chi tiêu {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} {CATEGORY} {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} chi tiêu {CATEGORY} {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} {CATEGORY} tuần này {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} {CATEGORY} {TIME} với {TIME} {SUFFIX}",
        "{PREFIX} cho xem {ACTION_TYPE} {CATEGORY} {TIME} {SUFFIX}",
        "{PREFIX} tổng hợp chi {TIME} {SUFFIX}",
        "{PREFIX} xem {ACTION_TYPE} chi tiêu {TIME} {SUFFIX}",
    ]
    
    # SET_LIMIT
    generators["SET_LIMIT"] = [
        "{PREFIX} {VERB} {TARGET} {CATEGORY} {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {AMOUNT} cho {CATEGORY} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {CATEGORY} {AMOUNT} {TIME} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {AMOUNT} {TIME} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {CATEGORY} {SUFFIX}",
    ]
    
    # SET_GOAL
    generators["SET_GOAL"] = [
        "{PREFIX} {VERB} {TARGET} {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} tiết kiệm {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} mua {CATEGORY} {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {AMOUNT} {TIME} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} tiết kiệm {AMOUNT} {TIME} {SUFFIX}",
    ]
    
    # ADD_GOAL
    generators["ADD_GOAL"] = [
        "{PREFIX} {VERB} thêm {AMOUNT} vào {TARGET} {SUFFIX}",
        "{PREFIX} {VERB} {AMOUNT} vào {TARGET} tiết kiệm {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} {TARGET} tiết kiệm thêm {AMOUNT} {SUFFIX}",
    ]
    
    # DELETE_RECORD
    generators["DELETE_RECORD"] = [
        "{PREFIX} {ACTION_TYPE} giao dịch vừa rồi {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} giao dịch {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} giao dịch {CATEGORY} vừa rồi {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} khoản chi {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} bill vừa nhập {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} gần nhất {SUFFIX}",
    ]
    
    # SET_TONE
    generators["SET_TONE"] = [
        "{PREFIX} {VERB} giọng nói sang {TONE_VAL} {SUFFIX}",
        "{PREFIX} {VERB} giọng Mimo thành {TONE_VAL} {SUFFIX}",
        "{PREFIX} nói chuyện kiểu {TONE_VAL} {SUFFIX}",
        "{PREFIX} đổi phong cách nói thành {TONE_VAL} {SUFFIX}",
        "{PREFIX} đổi giọng sang {TONE_VAL} {SUFFIX}",
    ]
    
    # SEARCH_RECORD
    generators["SEARCH_RECORD"] = [
        "{PREFIX} {ACTION_TYPE} giao dịch {CATEGORY} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} các giao dịch {CATEGORY} {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} khoản {CATEGORY} trên {AMOUNT} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} hóa đơn {CATEGORY} {TIME} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} từ khóa {CATEGORY} {SUFFIX}",
    ]
    
    # SUGGEST_BUDGET
    generators["SUGGEST_BUDGET"] = [
        "{PREFIX} gợi ý ngân sách {TIME} {SUFFIX}",
        "{PREFIX} gợi ý chi tiêu {TIME} {SUFFIX}",
        "{PREFIX} gợi ý tiết kiệm {TIME} {SUFFIX}",
        "{PREFIX} đề xuất hạn mức chi tiêu {TIME} {SUFFIX}",
        "{PREFIX} cho xin gợi ý chi tiêu {TIME} {SUFFIX}",
    ]
    
    # SYSTEM_SETTING
    generators["SYSTEM_SETTING"] = [
        "{PREFIX} mở màn hình cài đặt {SUFFIX}",
        "{PREFIX} chuyển sang {THEME_VAL} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} {THEME_VAL} trong app {SUFFIX}",
        "{PREFIX} tắt chế độ ban đêm {SUFFIX}",
        "{PREFIX} bật dark mode {SUFFIX}",
    ]
    
    # SET_USERNAME
    generators["SET_USERNAME"] = [
        "{PREFIX} gọi mình là {NAME} {SUFFIX}",
        "{PREFIX} gọi tôi là {NAME} {SUFFIX}",
        "{PREFIX} đổi tên hiển thị thành {NAME} {SUFFIX}",
        "{PREFIX} gọi tớ là {NAME} {SUFFIX}",
        "{PREFIX} tên mình là {NAME} {SUFFIX}",
    ]
    
    # SET_INCOME
    generators["SET_INCOME"] = [
        "{PREFIX} {VERB} thu nhập hàng tháng là {AMOUNT} {SUFFIX}",
        "{PREFIX} lương hàng tháng của mình là {AMOUNT} {SUFFIX}",
        "{PREFIX} {VERB} lương tháng này thành {AMOUNT} {SUFFIX}",
        "{PREFIX} thu nhập mỗi tháng là {AMOUNT} {SUFFIX}",
        "{PREFIX} cập nhật thu nhập là {AMOUNT} {SUFFIX}",
    ]
    
    # UPDATE_RECORD
    generators["UPDATE_RECORD"] = [
        "{PREFIX} sửa số tiền giao dịch {TIME} thành {AMOUNT} {SUFFIX}",
        "{PREFIX} đổi danh mục giao dịch {TIME} sang {CATEGORY} {SUFFIX}",
        "{PREFIX} sửa khoản chi vừa rồi thành {AMOUNT} {SUFFIX}",
        "{PREFIX} cập nhật giao dịch thành {AMOUNT} {SUFFIX}",
        "{PREFIX} đổi ghi chú thành {NAME} {SUFFIX}",
    ]
    
    # SET_ALERT
    generators["SET_ALERT"] = [
        "{PREFIX} bật cảnh báo hạn mức {CATEGORY} {SUFFIX}",
        "{PREFIX} tắt thông báo chi tiêu {CATEGORY} {SUFFIX}",
        "{PREFIX} đặt cảnh báo khi chi quá {AMOUNT} {SUFFIX}",
        "{PREFIX} {ACTION_TYPE} thông báo vượt hạn mức {CATEGORY} {SUFFIX}",
        "{PREFIX} bật cảnh báo chi tiêu {SUFFIX}",
    ]

    target_count = 20000
    
    print("Generating Action samples...")
    while len(existing_actions) < target_count:
        atype = random.choice(ACTION_TYPES)
        templates = generators[atype]
        tpl = random.choice(templates)
        
        # Select placeholder values
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)
        cat_str, _ = random.choice(CATEGORIES)
        time_str = random.choice(TIMES)
        amt_str = random.choice(AMOUNTS)
        
        # verbs and targets based on atype
        if atype == "ADD_GOAL":
            verb = random.choice(VERBS_ADD)
            target = random.choice(TARGETS_GOAL)
        elif atype == "SET_GOAL":
            verb = random.choice(VERBS_SET)
            target = random.choice(TARGETS_GOAL)
        elif atype == "SET_LIMIT":
            verb = random.choice(VERBS_SET + VERBS_ADD + VERBS_SUB)
            target = random.choice(TARGETS_LIMIT)
        else:
            verb = random.choice(VERBS_SET)
            target = random.choice(TARGETS_LIMIT)
            
        if atype == "SET_ALERT":
            action_type_val = random.choice(INDICATORS_SETTING)
        elif atype == "REPORT_GENERAL":
            action_type_val = random.choice(INDICATORS_REPORT)
        elif atype == "SEARCH_RECORD":
            action_type_val = random.choice(INDICATORS_SEARCH)
        elif atype == "DELETE_RECORD":
            action_type_val = random.choice(INDICATORS_DELETE)
        elif atype == "SYSTEM_SETTING":
            action_type_val = random.choice(INDICATORS_SETTING)
        else:
            action_type_val = "báo cáo"
            
        tone_val = random.choice(TONES)
        theme_val = random.choice(THEMES)
        name_val = random.choice(NAMES)
        
        # Build parameters
        params = {
            "PREFIX": (prefix, None),
            "SUFFIX": (suffix, None),
            "CATEGORY": (cat_str, None),
            "TIME": (time_str, None),
            "AMOUNT": (amt_str, None),
            "VERB": (verb, None),
            "TARGET": (target, None),
            "ACTION_TYPE": (action_type_val, None),
            "TONE_VAL": (tone_val, None),
            "THEME_VAL": (theme_val, None),
            "NAME": (name_val, None)
        }
        
        sample = build_ner_sample(tpl, params)
        txt = sample["text"]
        if txt and len(txt) > 3:
            norm_k = normalize_text_key(txt)
            if norm_k not in seen_action_texts:
                seen_action_texts.add(norm_k)
                existing_actions.append({"text": txt, "intent": "Action", "action_type": atype})

    # Save intent_action.csv
    with open(ACTION_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "action_type"])
        writer.writeheader()
        writer.writerows(existing_actions[:target_count])
    print(f"Saved {target_count} samples to {ACTION_CSV_PATH.name}")

    # Generate NER samples (Action + Record)
    print("Generating NER spans dataset...")
    
    # We will generate NER samples targeting labels: CATEGORY, AMOUNT, VERB, TARGET, TIME, ACTION_TYPE.
    # We mix both Action templates (using the generators list above) and Record templates.
    record_templates = [
        "{PREFIX} mua {CATEGORY} {AMOUNT} {TIME} {SUFFIX}",
        "{PREFIX} thanh toán tiền {CATEGORY} {AMOUNT} {TIME} {SUFFIX}",
        "{PREFIX} {CATEGORY} hết {AMOUNT} {TIME} {SUFFIX}",
        "{PREFIX} chi {AMOUNT} mua {CATEGORY} {TIME} {SUFFIX}",
        "{PREFIX} ăn {CATEGORY} {AMOUNT} {SUFFIX}",
        "{PREFIX} {CATEGORY} {AMOUNT} {SUFFIX}",
        "{PREFIX} lương tháng này về {AMOUNT} {SUFFIX}",
        "{PREFIX} thưởng KPI {AMOUNT} {SUFFIX}",
        "{PREFIX} bán hàng thu {AMOUNT} {SUFFIX}",
        "{PREFIX} mẹ cho {AMOUNT} {TIME} {SUFFIX}"
    ]
    
    while len(existing_ner) < target_count:
        # 50% Action, 50% Record
        is_action = random.random() < 0.5
        if is_action:
            atype = random.choice(ACTION_TYPES)
            templates = generators[atype]
            tpl = random.choice(templates)
            
            prefix = random.choice(PREFIXES)
            suffix = random.choice(SUFFIXES)
            cat_str, _ = random.choice(CATEGORIES)
            time_str = random.choice(TIMES)
            amt_str = random.choice(AMOUNTS)
            
            if atype == "ADD_GOAL":
                verb = random.choice(VERBS_ADD)
                target = random.choice(TARGETS_GOAL)
            elif atype == "SET_GOAL":
                verb = random.choice(VERBS_SET)
                target = random.choice(TARGETS_GOAL)
            elif atype == "SET_LIMIT":
                verb = random.choice(VERBS_SET + VERBS_ADD + VERBS_SUB)
                target = random.choice(TARGETS_LIMIT)
            else:
                verb = random.choice(VERBS_SET)
                target = random.choice(TARGETS_LIMIT)
                
            if atype == "SET_ALERT":
                action_type_val = random.choice(INDICATORS_SETTING)
            elif atype == "REPORT_GENERAL":
                action_type_val = random.choice(INDICATORS_REPORT)
            elif atype == "SEARCH_RECORD":
                action_type_val = random.choice(INDICATORS_SEARCH)
            elif atype == "DELETE_RECORD":
                action_type_val = random.choice(INDICATORS_DELETE)
            elif atype == "SYSTEM_SETTING":
                action_type_val = random.choice(INDICATORS_SETTING)
            else:
                action_type_val = None
                
            tone_val = random.choice(TONES)
            theme_val = random.choice(THEMES)
            name_val = random.choice(NAMES)
            
            # Map placeholders with standard NER entities
            params = {
                "PREFIX": (prefix, None),
                "SUFFIX": (suffix, None),
                "CATEGORY": (cat_str, "CATEGORY"),
                "TIME": (time_str, "TIME"),
                "AMOUNT": (amt_str, "AMOUNT"),
                "VERB": (verb, "VERB"),
                "TARGET": (target, "TARGET"),
                "TONE_VAL": (tone_val, None),
                "THEME_VAL": (theme_val, None),
                "NAME": (name_val, None)
            }
            if action_type_val:
                params["ACTION_TYPE"] = (action_type_val, "ACTION_TYPE")
        else:
            tpl = random.choice(record_templates)
            prefix = random.choice(PREFIXES)
            suffix = random.choice(SUFFIXES)
            cat_str, _ = random.choice(CATEGORIES)
            time_str = random.choice(TIMES)
            amt_str = random.choice(AMOUNTS)
            
            params = {
                "PREFIX": (prefix, None),
                "SUFFIX": (suffix, None),
                "CATEGORY": (cat_str, "CATEGORY"),
                "TIME": (time_str, "TIME"),
                "AMOUNT": (amt_str, "AMOUNT")
            }
            
        sample = build_ner_sample(tpl, params)
        txt = sample["text"]
        if txt and len(txt) > 3:
            norm_k = normalize_text_key(txt)
            if norm_k not in seen_ner_texts:
                seen_ner_texts.add(norm_k)
                existing_ner.append(sample)

    # Save ner_dataset.jsonl
    with open(NER_JSONL_PATH, "w", encoding="utf-8") as f:
        for item in existing_ner[:target_count]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {target_count} samples to {NER_JSONL_PATH.name}")
    print("Large dataset generation completed successfully!")

if __name__ == "__main__":
    main()
