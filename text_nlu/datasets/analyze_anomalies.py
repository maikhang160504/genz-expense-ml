import json

file_120 = r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\nlu_benchmark_120.json'
with open(file_120, 'r', encoding='utf-8') as f:
    data_120 = json.load(f)

anomalies = []

for i, item in enumerate(data_120):
    text = item['text'].lower()
    slots = item.get('slots', {})
    
    # 1. Check if goal_name is in text for SET_GOAL / ADD_GOAL
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
        f.write('Không tìm thấy bất thường.')
