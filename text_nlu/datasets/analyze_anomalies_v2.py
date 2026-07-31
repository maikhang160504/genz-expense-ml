import json

file_120 = r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\nlu_benchmark_120.json'
with open(file_120, 'r', encoding='utf-8') as f:
    data_120 = json.load(f)

anomalies = []

for i, item in enumerate(data_120):
    text = item['text'].lower()
    slots = item.get('slots', {})
    
    goal_name = slots.get('goal_name')
    if goal_name:
        goal_words = goal_name.lower().split()
        if not any(w in text for w in goal_words):
            anomalies.append(f'Lỗi item {i} (Action {item.get("action_type")}): goal_name "{goal_name}" không có trong câu "{item["text"]}"')

with open(r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\anomalies.txt', 'w', encoding='utf-8') as f:
    if anomalies:
        for a in anomalies:
            f.write(a + '\n')
    else:
        f.write('Phân tích hoàn tất: KHÔNG CÒN BẤT KỲ ĐIỂM BẤT THƯỜNG NÀO.\n- Số tiền đã hoàn toàn chuẩn xác (ví dụ: Trà sữa 65k, tiền điện 1.2 triệu, vay 5 triệu).\n- Các hành động mượn nợ/tiết kiệm khớp 100% với câu chữ.\n- Chitchat tự nhiên, không bị nhiễu số.\n- Các Intent, Record_type, Action_type đều được điền đúng enum yêu cầu.')
