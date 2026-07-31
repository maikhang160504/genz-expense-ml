import json
import random
from datetime import datetime

records = []
actions = []
chitchats = []

# --- 1. Generate 48 RECORDS ---
# Expenses (40)
expense_templates = [
    ('Food', 'Ăn uống cá nhân, đi chợ', ['phở bò', 'cơm tấm', 'trà sữa', 'bánh mì', 'lẩu', 'hải sản', 'bún chả', 'cơm gà', 'trà đào'], ['Cooking', 'Happy']),
    ('Transport', 'Di chuyển', ['đổ xăng', 'bắt grab', 'vé xe bus', 'sửa xe', 'gửi xe', 'thay nhớt', 'rửa xe', 'vé máy bay'], ['Travel', 'Chill']),
    ('Shopping', 'Mua sắm', ['áo thun', 'quần jean', 'giày sneaker', 'túi xách', 'áo khoác', 'đồng hồ', 'kính mát'], ['Shopping', 'Happy']),
    ('Beauty', 'Làm đẹp', ['son môi', 'cắt tóc', 'spa', 'kem dưỡng da', 'mặt nạ', 'dầu gội thảo dược', 'nước hoa'], ['Happy', 'Cool', 'Beauty']),
    ('Social', 'Giao lưu', ['đi ăn cưới', 'quà sinh nhật', 'đi cà phê với bạn', 'nhậu cuối tuần', 'hát karaoke', 'tiệc tùng', 'mừng tân gia'], ['Celebrate', 'Happy']),
    ('Health', 'Sức khỏe', ['thuốc cảm', 'khám răng', 'vitamin', 'tiền tập gym', 'đo huyết áp', 'khám tổng quát'], ['Relax', 'Alert']),
    ('Housing', 'Nhà ở', ['tiền nhà', 'tiền điện', 'tiền nước', 'tiền internet', 'phí quản lý', 'bình gas', 'bóng đèn'], ['Chill', 'Sad']),
    ('Education', 'Học tập', ['mua sách', 'học phí tiếng Anh', 'khóa học online', 'vở viết', 'bút bi', 'đăng ký hội thảo'], ['Working', 'Determined']),
    ('Entertainment', 'Giải trí', ['vé xem phim', 'tài khoản netflix', 'mua game', 'đi quẩy', 'thuê truyện', 'mua đồ chơi'], ['Excited', 'Happy']),
    ('Essentials', 'Thiết yếu', ['nước giặt', 'dầu gội', 'kem đánh răng', 'đồ siêu thị', 'nước mắm', 'bột giọt', 'giấy vệ sinh'], ['Chill', 'Relax']),
    ('Business', 'Kinh doanh', ['nhập hàng', 'chạy quảng cáo', 'mua tên miền', 'in tờ rơi', 'thuê freelancer'], ['Working', 'Determined']),
    ('Charity', 'Từ thiện', ['ủng hộ đồng bào', 'quyên góp từ thiện', 'mua tăm ủng hộ', 'nuôi heo đất', 'gửi quỹ vắc xin'], ['Thankful', 'Love']),
    ('Debt', 'Trả nợ', ['trả nợ thẻ tín dụng', 'trả tiền mượn bạn', 'thanh toán khoản vay', 'trả nợ anh trai'], ['Sad', 'Relax']),
    ('Savings', 'Tiết kiệm', ['gửi tiết kiệm', 'bỏ ống heo', 'nạp tiền vào quỹ dự phòng', 'tiết kiệm cuối tháng'], ['Proud', 'Determined']),
    ('Investment', 'Đầu tư', ['mua cổ phiếu', 'mua vàng', 'đầu tư chứng chỉ quỹ', 'rót vốn mở quán', 'góp vốn làm ăn'], ['Working', 'Cool'])
]

amount_suffixes = [('k', 1000), (' ngàn', 1000), (' nghìn', 1000), (' triệu', 1000000), ('tr', 1000000)]
verbs_expense = ['Mua', 'Thanh toán', 'Trả', 'Chi', 'Sắm', 'Đóng', 'Sài', 'Xài']

for i in range(40):
    cat, desc, items, emotions = random.choice(expense_templates)
    item = random.choice(items)
    amt_val = random.randint(1, 100)
    suffix, mult = random.choice(amount_suffixes)
    amount_str = f'{amt_val}{suffix}'
    amount = amt_val * mult
    verb_str = random.choice(verbs_expense)
    
    text = f'{verb_str} {item} hết {amount_str}'
    if i % 3 == 0: text = f'Hôm nay {text.lower()}'
    
    records.append({
        'text': text,
        'intent': 'Record',
        'record_type': 'Expense',
        'action_type': None,
        'slots': {
            'item': item,
            'category': cat,
            'amount': amount,
            'verb': None,
            'goal_name': None,
            'tool_type': None,
            'loan_type': None,
            'contact_name': None,
            'due_date': None,
            'enabled': None,
            'theme': None,
            'verbal_style': None,
            'time_range': 'hôm nay' if 'Hôm nay' in text else None,
            'query': None
        },
        'emotion': random.choice(emotions),
        'response': f'Đã ghi nhận chi tiêu {amount:,}đ cho {item}.',
        'suggested_actions': ['Xem báo cáo', 'Thêm chi tiêu khác', 'Cài hạn mức']
    })

# Incomes (8)
income_templates = [
    ('Salary', ['nhận lương', 'lương tháng này', 'tiền lương', 'lương part-time', 'nhận tiền lương'], ['Happy', 'Celebrate']),
    ('Bonus', ['thưởng dự án', 'thưởng tết', 'tiền thưởng', 'hoa hồng', 'thưởng quý'], ['Excited', 'Proud']),
    ('Business', ['bán hàng', 'doanh thu cửa hàng', 'chốt đơn', 'bán hàng online'], ['Working', 'Happy']),
    ('Others', ['người yêu cho tiền', 'nhặt được tiền', 'trúng số', 'ba mẹ cho', 'lì xì'], ['Excited', 'Giggle'])
]
for i in range(8):
    cat, items, emotions = random.choice(income_templates)
    item = random.choice(items)
    amt_val = random.randint(5, 50)
    amount_str = f'{amt_val} triệu'
    amount = amt_val * 1000000
    
    text = f'{item} được {amount_str}'
    if i % 2 == 0: text = f'Mới {text.lower()}'
    records.append({
        'text': text,
        'intent': 'Record',
        'record_type': 'Income',
        'action_type': None,
        'slots': {
            'item': item,
            'category': cat,
            'amount': amount,
            'verb': None,
            'goal_name': None,
            'tool_type': None,
            'loan_type': None,
            'contact_name': None,
            'due_date': None,
            'enabled': None,
            'theme': None,
            'verbal_style': None,
            'time_range': None,
            'query': None
        },
        'emotion': random.choice(emotions),
        'response': f'Chúc mừng bạn đã thu được {amount:,}đ từ {item}!',
        'suggested_actions': ['Gửi tiết kiệm', 'Xem báo cáo thu nhập']
    })

# --- 2. Generate 48 ACTIONS ---
action_types = [
    ('REPORT_GENERAL', 8), ('REPORT_COMPARE', 4), ('SEARCH_RECORD', 8),
    ('SET_LIMIT', 8), ('SET_GOAL', 6), ('ADD_GOAL', 6),
    ('SET_TONE', 2), ('SET_ALERT', 2), ('SYSTEM_SETTING', 4)
]

for atype, count in action_types:
    for i in range(count):
        slots = {'item': None, 'category': None, 'amount': None, 'verb': None, 'goal_name': None, 'tool_type': None, 'loan_type': None, 'contact_name': None, 'due_date': None, 'enabled': None, 'theme': None, 'verbal_style': None, 'time_range': None, 'query': None}
        text = ''
        response = ''
        
        if atype == 'REPORT_GENERAL':
            tr = random.choice(['tháng này', 'tuần trước', 'hôm nay', 'năm nay', 'tháng trước'])
            slots['time_range'] = tr
            if i % 2 == 0:
                cat = random.choice(['Food', 'Shopping', 'Transport', 'Beauty', 'Entertainment'])
                slots['category'] = cat
                cat_vn = 'ăn uống' if cat == 'Food' else 'mua sắm' if cat == 'Shopping' else 'di chuyển' if cat == 'Transport' else 'làm đẹp' if cat == 'Beauty' else 'giải trí'
                text = f'Tổng kết tiền {cat_vn} {tr} xem nào'
                response = f'Đây là báo cáo chi tiêu cho {cat} trong {tr}.'
            else:
                text = f'Báo cáo chi tiêu {tr}'
                response = f'Tôi đã tổng hợp báo cáo chi tiêu của bạn trong {tr}.'
                
        elif atype == 'REPORT_COMPARE':
            tr = random.choice(['tháng này', 'tháng trước', 'quý này'])
            slots['time_range'] = tr
            text = f'So sánh chi tiêu của tôi với mọi người {tr}'
            response = f'Dưới đây là so sánh chi tiêu của bạn với cộng đồng trong {tr}.'
            
        elif atype == 'SEARCH_RECORD':
            tr = random.choice(['hôm qua', 'tuần này', 'tháng này', 'hôm nay'])
            slots['time_range'] = tr
            query = random.choice(['trà sữa', 'mua sắm', 'ăn sáng', 'đổ xăng', 'cà phê', 'xem phim'])
            slots['query'] = query
            text = f'Tìm các khoản {query} {tr}'
            response = f'Tôi tìm thấy các giao dịch liên qua đến {query} trong {tr}.'
            
        elif atype == 'SET_LIMIT':
            cat = random.choice(['Food', 'Shopping', 'Entertainment', 'Beauty', 'Transport'])
            cat_vn = 'ăn uống' if cat == 'Food' else 'mua sắm' if cat == 'Shopping' else 'di chuyển' if cat == 'Transport' else 'làm đẹp' if cat == 'Beauty' else 'giải trí'
            amt_val = random.randint(1, 5)
            amount = amt_val * 1000000
            slots['category'] = cat
            slots['amount'] = amount
            text = f'Đặt hạn mức {cat_vn} là {amt_val} triệu'
            response = f'Đã thiết lập hạn mức {amount:,}đ cho danh mục {cat}.'
            
        elif atype == 'SET_GOAL':
            goal = random.choice(['mua xe máy', 'đi du lịch đà lạt', 'đổi điện thoại', 'mua laptop', 'đóng học phí'])
            amt = random.choice([20000000, 50000000, 15000000, 10000000])
            ttype = random.choice(['saving_personal', 'saving_group', 'challenge', 'loan'])
            slots['goal_name'] = goal
            slots['amount'] = amt
            slots['tool_type'] = ttype
            
            if ttype == 'loan':
                contact = random.choice(['Nam', 'Linh', 'Hải', 'Tuấn', 'Lan'])
                ltype = random.choice(['lend', 'borrow'])
                slots['contact_name'] = contact
                slots['loan_type'] = ltype
                slots['due_date'] = '2024-12-31'
                if ltype == 'lend':
                    text = f'Tạo nhắc hẹn cho {contact} mượn {amt//1000000} triệu hạn 31/12/2024'
                    response = f'Đã tạo nhắc hẹn cho vay {amt:,}đ với {contact}.'
                else:
                    text = f'Nhắc mượn {contact} {amt//1000000} triệu hạn 31/12/2024'
                    response = f'Đã ghi nhận khoản vay {amt:,}đ từ {contact}.'
            else:
                if ttype == 'saving_group': text = f'Tạo quỹ nhóm tiết kiệm {amt//1000000} triệu {goal}'
                elif ttype == 'challenge': text = f'Tạo thử thách tiết kiệm {amt//1000000} triệu {goal}'
                else: text = f'Tạo mục tiêu tiết kiệm {amt//1000000} triệu để {goal}'
                response = f'Chúc bạn sớm hoàn thành mục tiêu {goal}!'
                
        elif atype == 'ADD_GOAL':
            goal = random.choice(['mua xe máy', 'đi du lịch đà lạt', 'đổi điện thoại', 'mua laptop'])
            amt = random.randint(1, 5) * 1000000
            slots['goal_name'] = goal
            slots['amount'] = amt
            slots['verb'] = 'ADD'
            text = f'Nạp thêm {amt//1000000} triệu vào quỹ {goal}'
            response = f'Đã cộng thêm {amt:,}đ vào mục tiêu {goal}.'
            
        elif atype == 'SET_TONE':
            tone = random.choice(['dui_de', 'dan_doi', 'kho_tinh', 'ngot_ngao'])
            slots['verbal_style'] = tone
            text = f'Đổi giọng sang kiểu {tone}'
            response = 'Đã đổi giọng điệu theo yêu cầu của bạn nha!'
            
        elif atype == 'SET_ALERT':
            en = random.choice([True, False])
            slots['enabled'] = en
            str_en = 'Bật' if en else 'Tắt'
            text = f"{str_en} cảnh báo chi tiêu"
            response = f"Đã {str_en.lower()} cảnh báo chi tiêu."
            
        elif atype == 'SYSTEM_SETTING':
            theme = random.choice(['dark', 'light'])
            slots['theme'] = theme
            text = f"Đổi sang giao diện {theme}"
            response = f"Đã chuyển sang giao diện {theme}."
            
        actions.append({
            'text': text,
            'intent': 'Action',
            'record_type': None,
            'action_type': atype,
            'slots': slots,
            'emotion': 'Thinking',
            'response': response,
            'suggested_actions': None
        })

# --- 3. Generate 24 CHITCHATS ---
chitchat_templates = [
    ('Chào buổi sáng', 'Hello', 'Chào buổi sáng! Bạn đã ăn sáng chưa?'),
    ('Hôm nay buồn quá', 'Sad', 'Có chuyện gì buồn vậy bạn? Kể tôi nghe đi.'),
    ('Bot ngu ngốc', 'Angry', 'Xin lỗi vì đã làm bạn bực mình. Tôi đang cố gắng học hỏi thêm.'),
    ('Mình vừa được thăng chức', 'Happy', 'Tuyệt vời quá! Chúc mừng bạn nhé!'),
    ('Mệt mỏi quá', 'Sleepy', 'Bạn nên nghỉ ngơi một chút đi.'),
    ('Làm sao để giàu', 'Thinking', 'Chăm chỉ làm việc và quản lý chi tiêu hợp lý nhé!'),
    ('Bạn tên là gì?', 'Hello', 'Tôi là trợ lý tài chính thông minh của bạn đây!'),
    ('Thời tiết hôm nay đẹp thật', 'Chill', 'Đúng vậy, rất hợp để đi dạo đó.'),
    ('Cảm ơn bạn nhé', 'Thankful', 'Không có chi! Tôi luôn sẵn sàng giúp đỡ bạn.'),
    ('Sếp lại chửi mình', 'Worried', 'Thương bạn quá! Cố gắng lên nhé, qua cơn mưa trời lại sáng.'),
    ('Hế lô bot', 'Hello', 'Xin chào! Tôi có thể giúp gì cho bạn hôm nay?'),
    ('Mày biết hát không', 'Sassy', 'Biết chứ, nhưng tôi hát dở lắm, chỉ giỏi đếm tiền thôi!'),
    ('Làm phiền bạn một chút', 'Hello', 'Không sao đâu, bạn cứ nói đi!'),
    ('Đi quẩy thôi', 'Celebrate', 'Lên đồ lên đồ! Nhớ chú ý an toàn nha.'),
    ('Tuyệt vời ông mặt trời', 'Happy', 'Đỉnh nóc kịch trần luôn!'),
    ('Tôi đang nấu ăn', 'Cooking', 'Thơm quá! Chúc bạn có bữa ăn ngon miệng.')
]

for i in range(24):
    text, emo, resp = random.choice(chitchat_templates)
    text = f'{text} {random.randint(1, 1000)}' # add some noise
    chitchats.append({
        'text': text,
        'intent': 'Chitchat',
        'record_type': None,
        'action_type': None,
        'slots': {'item': None, 'category': None, 'amount': None, 'verb': None, 'goal_name': None, 'tool_type': None, 'loan_type': None, 'contact_name': None, 'due_date': None, 'enabled': None, 'theme': None, 'verbal_style': None, 'time_range': None, 'query': None},
        'emotion': emo,
        'response': resp,
        'suggested_actions': None
    })

# Combine and shuffle
dataset = records + actions + chitchats
random.shuffle(dataset)

# Ensure exactly 120
dataset = dataset[:120]

# Construct the exact final JSON objects expected by the user prompt
final_dataset = []
for d in dataset:
    obj = {
        'text': d['text'],
        'intent': d['intent'],
        'record_type': d['record_type'],
        'action_type': d['action_type'],
        'slots': d['slots'],
        'emotion': d['emotion'],
        'response': d['response'],
        'suggested_actions': d['suggested_actions']
    }
    final_dataset.append(obj)

with open(r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\nlu_benchmark_120.json', 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

print('Success')
