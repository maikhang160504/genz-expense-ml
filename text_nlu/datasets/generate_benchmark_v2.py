import json

# Format: (text, category, amount)
records_expense = [
    ("Mua ly trà sữa The Alley 65k", "Food", 65000),
    ("Ăn trưa cơm tấm sườn bì 50 ngàn", "Food", 50000),
    ("Đi chợ mua thức ăn hết 300k", "Food", 300000),
    ("Đổ xăng đầy bình 80k", "Transport", 80000),
    ("Bắt grab đi làm 45 ngàn", "Transport", 45000),
    ("Thay nhớt xe máy 120k", "Transport", 120000),
    ("Sắm cái áo khoác trên Shopee 250k", "Shopping", 250000),
    ("Mua đôi giày sneaker 1.5 triệu", "Shopping", 1500000),
    ("Tiền điện tháng này 1 triệu 2", "Housing", 1200000),
    ("Đóng tiền trọ 3 triệu", "Housing", 3000000),
    ("Đóng tiền nước 150 ngàn", "Housing", 150000),
    ("Mua chai dầu gội đầu 180k", "Essentials", 180000),
    ("Đi siêu thị mua kem đánh răng 45 ngàn", "Essentials", 45000),
    ("Nước giặt quần áo 200k", "Essentials", 200000),
    ("Cắt tóc nam 60 ngàn", "Beauty", 60000),
    ("Mua thỏi son 400k tặng mình", "Beauty", 400000),
    ("Đi spa chăm sóc da 1 triệu", "Beauty", 1000000),
    ("Mua thuốc cảm 100k", "Health", 100000),
    ("Khám răng hết 500 ngàn", "Health", 500000),
    ("Đóng tiền tập gym 3 tháng 1 triệu 5", "Health", 1500000),
    ("Đi nhậu với đồng nghiệp 400k", "Social", 400000),
    ("Mừng cưới bạn thân 1 triệu", "Social", 1000000),
    ("Mua quà sinh nhật cho crush 500 ngàn", "Social", 500000),
    ("Học phí lớp tiếng anh 4 triệu", "Education", 4000000),
    ("Mua sách lập trình 250k", "Education", 250000),
    ("Mua khóa học trên Udemy 300 ngàn", "Education", 300000),
    ("Gia hạn Netflix 260k", "Entertainment", 260000),
    ("Đi xem phim chiếu rạp 150 ngàn", "Entertainment", 150000),
    ("Mua nạp thẻ game 100k", "Entertainment", 100000),
    ("Ủng hộ quỹ vaccine 200 ngàn", "Charity", 200000),
    ("Quyên góp đồng bào lũ lụt 500k", "Charity", 500000),
    ("Gửi tiền tiết kiệm 5 triệu", "Savings", 5000000),
    ("Bỏ ống heo 50 ngàn", "Savings", 50000),
    ("Mua 1 chỉ vàng 8 triệu", "Investment", 8000000),
    ("Đầu tư chứng khoán 10 triệu", "Investment", 10000000),
    ("Trả nợ thẻ tín dụng 3 triệu", "Debt", 3000000),
    ("Trả tiền mượn Hùng hôm bữa 200k", "Debt", 200000),
    ("Nhập hàng để bán 5 triệu", "Business", 5000000),
    ("Chạy quảng cáo Facebook 1 triệu", "Business", 1000000),
    ("Chi tiêu lặt vặt 50k", "Others", 50000)
]

records_income = [
    ("Nhận lương tháng mười 15 triệu", "Salary", 15000000),
    ("Lương làm thêm 2 triệu", "Salary", 2000000),
    ("Được thưởng dự án 5 triệu", "Bonus", 5000000),
    ("Thưởng lễ 1 triệu", "Bonus", 1000000),
    ("Chốt đơn hàng 3 triệu", "Business", 3000000),
    ("Lãi tiết kiệm 200k", "Business", 200000),
    ("Trúng số 100 ngàn", "Bonus", 100000),
    ("Được mẹ cho 500k", "Bonus", 500000)
]

# (text, action_type, category, amount, time_range, query, goal_name, tool_type, contact_name, loan_type, due_date, enabled, theme, verbal_style)
actions_list = [
    # REPORT_GENERAL (8)
    ("Tháng này tiêu bao nhiêu tiền rồi?", "REPORT_GENERAL", None, None, "tháng này", None, None, None, None, None, None, None, None, None),
    ("Tổng kết tiền ăn uống hôm nay", "REPORT_GENERAL", "Food", None, "hôm nay", None, None, None, None, None, None, None, None, None),
    ("Báo cáo chi tiêu tuần trước", "REPORT_GENERAL", None, None, "tuần trước", None, None, None, None, None, None, None, None, None),
    ("Thống kê khoản mua sắm trong tháng trước", "REPORT_GENERAL", "Shopping", None, "tháng trước", None, None, None, None, None, None, None, None, None),
    ("Năm nay tiêu bao nhiêu rồi bot", "REPORT_GENERAL", None, None, "năm nay", None, None, None, None, None, None, None, None, None),
    ("Xem chi phí di chuyển quý này", "REPORT_GENERAL", "Transport", None, "quý này", None, None, None, None, None, None, None, None, None),
    ("Báo cáo tiền điện nước tháng này coi", "REPORT_GENERAL", "Housing", None, "tháng này", None, None, None, None, None, None, None, None, None),
    ("Hôm qua xài hết nhiêu tiền?", "REPORT_GENERAL", None, None, "hôm qua", None, None, None, None, None, None, None, None, None),

    # REPORT_COMPARE (4)
    ("So sánh mức chi tiêu tháng này với cộng đồng", "REPORT_COMPARE", None, None, "tháng này", None, None, None, None, None, None, None, None, None),
    ("Tôi tiêu nhiều hơn mọi người trong tuần này không?", "REPORT_COMPARE", None, None, "tuần này", None, None, None, None, None, None, None, None, None),
    ("Tháng trước ăn uống có tốn hơn người khác không?", "REPORT_COMPARE", "Food", None, "tháng trước", None, None, None, None, None, None, None, None, None),
    ("Đối chiếu chi tiêu của tôi năm nay", "REPORT_COMPARE", None, None, "năm nay", None, None, None, None, None, None, None, None, None),

    # SEARCH_RECORD (8)
    ("Tìm các khoản mua trà sữa tháng này", "SEARCH_RECORD", "Food", None, "tháng này", "trà sữa", None, None, None, None, None, None, None, None),
    ("Liệt kê giao dịch đổ xăng hôm qua", "SEARCH_RECORD", "Transport", None, "hôm qua", "đổ xăng", None, None, None, None, None, None, None, None),
    ("Tuần trước có mua cái áo nào không?", "SEARCH_RECORD", "Shopping", None, "tuần trước", "áo", None, None, None, None, None, None, None, None),
    ("Tra cứu tiền điện nước năm nay", "SEARCH_RECORD", "Housing", None, "năm nay", "tiền điện nước", None, None, None, None, None, None, None, None),
    ("Tìm hóa đơn siêu thị hôm nay", "SEARCH_RECORD", "Essentials", None, "hôm nay", "siêu thị", None, None, None, None, None, None, None, None),
    ("Tôi đã chi bao nhiêu cho sinh nhật tháng này", "SEARCH_RECORD", "Social", None, "tháng này", "sinh nhật", None, None, None, None, None, None, None, None),
    ("Lọc các giao dịch netflix", "SEARCH_RECORD", "Entertainment", None, None, "netflix", None, None, None, None, None, None, None, None),
    ("Tìm khoản mua thuốc cảm hôm thứ hai", "SEARCH_RECORD", "Health", None, "hôm thứ hai", "thuốc cảm", None, None, None, None, None, None, None, None),

    # SET_LIMIT (8)
    ("Đặt hạn mức ăn uống 3 triệu", "SET_LIMIT", "Food", 3000000, None, None, None, None, None, None, None, None, None, None),
    ("Cài hạn mức mua sắm tháng này là 2 triệu", "SET_LIMIT", "Shopping", 2000000, None, None, None, None, None, None, None, None, None, None),
    ("Giới hạn tiền di chuyển 500k thôi", "SET_LIMIT", "Transport", 500000, None, None, None, None, None, None, None, None, None, None),
    ("Set quota giải trí 1 triệu", "SET_LIMIT", "Entertainment", 1000000, None, None, None, None, None, None, None, None, None, None),
    ("Hạn mức thiết yếu để 1 triệu rưỡi", "SET_LIMIT", "Essentials", 1500000, None, None, None, None, None, None, None, None, None, None),
    ("Đặt ngân sách làm đẹp 800 ngàn", "SET_LIMIT", "Beauty", 800000, None, None, None, None, None, None, None, None, None, None),
    ("Khóa học giáo dục hạn mức 5 triệu", "SET_LIMIT", "Education", 5000000, None, None, None, None, None, None, None, None, None, None),
    ("Cài giới hạn giao lưu bạn bè 2 triệu", "SET_LIMIT", "Social", 2000000, None, None, None, None, None, None, None, None, None, None),

    # SET_GOAL (6)
    ("Tạo mục tiêu tiết kiệm 30 triệu để mua xe máy", "SET_GOAL", None, 30000000, None, None, "mua xe máy", "saving_personal", None, None, None, None, None, None),
    ("Lập quỹ nhóm 50 triệu đi du lịch Thái Lan", "SET_GOAL", None, 50000000, None, None, "đi du lịch Thái Lan", "saving_group", None, None, None, None, None, None),
    ("Tạo thử thách tiết kiệm 10 triệu trong 3 tháng", "SET_GOAL", None, 10000000, None, None, "tiết kiệm 10 triệu", "challenge", None, None, None, None, None, None),
    ("Tạo nhắc hẹn cho Nam mượn 5 triệu hạn 15/08", "SET_GOAL", None, 5000000, None, None, None, "loan", "Nam", "lend", "2026-08-15", None, None, None),
    ("Nhắc vay Linh 2 triệu hạn 31/12/2026", "SET_GOAL", None, 2000000, None, None, None, "loan", "Linh", "borrow", "2026-12-31", None, None, None),
    ("Mở thử thách nhóm tiết kiệm 20 triệu mua laptop", "SET_GOAL", None, 20000000, None, None, "mua laptop", "challenge_group", None, None, None, None, None, None),

    # ADD_GOAL (6)
    ("Nạp 500k vào quỹ mua xe", "ADD_GOAL", None, 500000, None, None, "mua xe", None, None, None, None, None, None, None),
    ("Chuyển thêm 2 triệu vào heo đất đi du lịch", "ADD_GOAL", None, 2000000, None, None, "đi du lịch", None, None, None, None, None, None, None),
    ("Cộng 1 triệu cho mục tiêu mua điện thoại", "ADD_GOAL", None, 1000000, None, None, "mua điện thoại", None, None, None, None, None, None, None),
    ("Bỏ vô quỹ tiết kiệm cá nhân 300 ngàn", "ADD_GOAL", None, 300000, None, None, "tiết kiệm cá nhân", None, None, None, None, None, None, None),
    ("Đóng 500k vô quỹ nhóm đi Đà Lạt", "ADD_GOAL", None, 500000, None, None, "quỹ nhóm đi Đà Lạt", None, None, None, None, None, None, None),
    ("Thêm 100k vào mục tiêu mua giày mới", "ADD_GOAL", None, 100000, None, None, "mua giày mới", None, None, None, None, None, None, None),

    # SET_TONE (2)
    ("Đổi giọng điệu sang vui vẻ dễ thương nhé", "SET_TONE", None, None, None, None, None, None, None, None, None, None, None, "dui_de"),
    ("Chuyển qua nói chuyện nghiêm túc khó tính đi", "SET_TONE", None, None, None, None, None, None, None, None, None, None, None, "kho_tinh"),

    # SET_ALERT (2)
    ("Bật cảnh báo chi tiêu lên giúp tôi", "SET_ALERT", None, None, None, None, None, None, None, None, None, True, None, None),
    ("Tắt cái thông báo vượt hạn mức đi", "SET_ALERT", None, None, None, None, None, None, None, None, None, False, None, None),

    # SYSTEM_SETTING (4)
    ("Chuyển sang giao diện nền tối", "SYSTEM_SETTING", None, None, None, None, None, None, None, None, None, None, "dark", None),
    ("Đổi app sang màu sáng đi", "SYSTEM_SETTING", None, None, None, None, None, None, None, None, None, None, "light", None),
    ("Bật dark mode nha", "SYSTEM_SETTING", None, None, None, None, None, None, None, None, None, None, "dark", None),
    ("Trở lại giao diện sáng sủa đi", "SYSTEM_SETTING", None, None, None, None, None, None, None, None, None, None, "light", None)
]

# (text, emotion)
chitchats_list = [
    ("Chào buổi sáng nha bot!", "Hello"),
    ("Hôm nay tao mệt quá, đi làm chán ghê.", "Sad"),
    ("Mày tư vấn sai bét rồi!", "Angry"),
    ("Mới được sếp tăng lương, vui quá trời!", "Happy"),
    ("Cảm ơn mày nhiều nha.", "Thankful"),
    ("Làm sao để tiết kiệm tiền hiệu quả vậy?", "Thinking"),
    ("Mày ăn cơm chưa?", "Hello"),
    ("Tối nay quẩy thôi anh em ơi!", "Celebrate"),
    ("Buồn ngủ quá bot ơi.", "Sleepy"),
    ("Đang thư giãn ở quán cà phê chill chill", "Chill"),
    ("Sắp đi du lịch Thái Lan rồi háo hức quá", "Travel"),
    ("Mày biết nói đùa không?", "Sassy"),
    ("Bot ngu ngốc", "Error"),
    ("Tuyệt vời, tao vừa hoàn thành deadline!", "Success"),
    ("Trời ơi tao tiêu lố tay vào shopee rồi", "Worried"),
    ("Đang nấu ăn ngon lắm nè.", "Cooking"),
    ("Lêu lêu đồ con bot ngốc", "Taunting"),
    ("Hôm nay tao rất quyết tâm học code!", "Determined"),
    ("Đang đi shopping sắm đồ tết.", "Shopping"),
    ("Ngầu chưa nè bot?", "Cool"),
    ("Đang tập trung làm việc cày cuốc.", "Working"),
    ("Tự hào về bản thân vì đã để dành được 10 triệu", "Proud"),
    ("Tao lỡ tay xóa nhầm rồi, xin lỗi nha", "Sorry"),
    ("Yêu con bot này ghê", "Love")
]

final_data = []

# Generate Record json
for text, cat, amt in records_expense:
    obj = {
        "text": text,
        "intent": "Record",
        "record_type": "Expense",
        "action_type": None,
        "slots": {
            "item": text.split()[1] if len(text.split()) > 1 else text,
            "category": cat,
            "amount": amt,
            "verb": None,
            "goal_name": None,
            "tool_type": None,
            "loan_type": None,
            "contact_name": None,
            "due_date": None,
            "enabled": None,
            "theme": None,
            "verbal_style": None,
            "time_range": None,
            "query": None
        },
        "emotion": "Happy",
        "response": f"Đã ghi nhận chi tiêu {amt:,}đ. Hãy kiểm soát tài chính tốt nhé!",
        "suggested_actions": ["Xem báo cáo", "Cài hạn mức"]
    }
    final_data.append(obj)

for text, cat, amt in records_income:
    obj = {
        "text": text,
        "intent": "Record",
        "record_type": "Income",
        "action_type": None,
        "slots": {
            "item": text.split()[1] if len(text.split()) > 1 else text,
            "category": cat,
            "amount": amt,
            "verb": None,
            "goal_name": None,
            "tool_type": None,
            "loan_type": None,
            "contact_name": None,
            "due_date": None,
            "enabled": None,
            "theme": None,
            "verbal_style": None,
            "time_range": None,
            "query": None
        },
        "emotion": "Happy",
        "response": f"Chúc mừng bạn đã thu được khoản tiền {amt:,}đ!",
        "suggested_actions": ["Tạo tiết kiệm", "Xem báo cáo"]
    }
    final_data.append(obj)

# Generate Action json
for text, atype, cat, amt, time_range, query, goal_name, tool_type, contact_name, loan_type, due_date, enabled, theme, verbal_style in actions_list:
    obj = {
        "text": text,
        "intent": "Action",
        "record_type": None,
        "action_type": atype,
        "slots": {
            "item": None,
            "category": cat,
            "amount": amt,
            "verb": None,
            "goal_name": goal_name,
            "tool_type": tool_type,
            "loan_type": loan_type,
            "contact_name": contact_name,
            "due_date": due_date,
            "enabled": enabled,
            "theme": theme,
            "verbal_style": verbal_style,
            "time_range": time_range,
            "query": query
        },
        "emotion": "Thinking",
        "response": "Đã thực hiện yêu cầu của bạn.",
        "suggested_actions": None
    }
    final_data.append(obj)

# Generate Chitchat json
for text, emo in chitchats_list:
    obj = {
        "text": text,
        "intent": "Chitchat",
        "record_type": None,
        "action_type": None,
        "slots": {
            "item": None, "category": None, "amount": None, "verb": None, "goal_name": None, "tool_type": None, "loan_type": None, "contact_name": None, "due_date": None, "enabled": None, "theme": None, "verbal_style": None, "time_range": None, "query": None
        },
        "emotion": emo,
        "response": "Chatbot đang lắng nghe và trò chuyện với bạn đây!",
        "suggested_actions": None
    }
    final_data.append(obj)

import random
# Just shuffle a bit
# random.seed(42)
# random.shuffle(final_data)

with open(r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\nlu_benchmark_120.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

# Also generate nlu_benchmark.jsonl
with open(r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\nlu_benchmark.jsonl', 'w', encoding='utf-8') as f:
    for item in final_data:
        cat = item['slots'].get('category')
        if not cat: cat = 'None'
        
        rec = item.get('record_type')
        if not rec: rec = 'None'
        else: rec = str(rec).lower()
        
        act = item.get('action_type')
        if not act: act = 'None'
        
        out_obj = {
            'text': item.get('text', ''),
            'expected_intent': item.get('intent', 'None'),
            'expected_category': cat,
            'expected_record_type': rec,
            'expected_action_type': act
        }
        f.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

print(f'Generated {len(final_data)} items perfectly!')
