import json

file_path = r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\nlu_benchmark.jsonl'
anomalies = []

valid_expense_cats = ["Food", "Transport", "Shopping", "Beauty", "Social", "Health", "Housing", "Education", "Entertainment", "Essentials", "Business", "Charity", "Debt", "Savings", "Investment", "Others"]
valid_income_cats = ["Salary", "Bonus", "Business", "Others"]
valid_actions = ["REPORT_GENERAL", "REPORT_COMPARE", "SET_LIMIT", "SET_GOAL", "ADD_GOAL", "SET_TONE", "SEARCH_RECORD", "SUGGEST_BUDGET", "SYSTEM_SETTING", "SET_USERNAME", "SET_ALERT"]

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if not line.strip(): continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            anomalies.append(f"Dòng {i}: Lỗi cú pháp JSON.")
            continue
            
        text = data.get('text', '')
        intent = data.get('expected_intent')
        cat = data.get('expected_category')
        rec = data.get('expected_record_type')
        act = data.get('expected_action_type')
        
        if intent not in ['Record', 'Action', 'Chitchat']:
            anomalies.append(f'Dòng {i}: Intent không hợp lệ ("{intent}")')
            
        if intent == 'Record':
            if rec not in ['income', 'expense']:
                anomalies.append(f'Dòng {i}: Record nhưng record_type ("{rec}") không phải income/expense.')
            if rec == 'expense' and cat not in valid_expense_cats:
                anomalies.append(f'Dòng {i}: Record Expense nhưng category ("{cat}") không hợp lệ.')
            if rec == 'income' and cat not in valid_income_cats:
                anomalies.append(f'Dòng {i}: Record Income nhưng category ("{cat}") không hợp lệ.')
            if act != 'None':
                anomalies.append(f'Dòng {i}: Record nhưng action_type không phải None ("{act}").')
                
        elif intent == 'Action':
            if act not in valid_actions:
                anomalies.append(f'Dòng {i}: Action nhưng action_type ("{act}") không hợp lệ.')
            if rec != 'None':
                anomalies.append(f'Dòng {i}: Action nhưng record_type không phải None ("{rec}").')
                
            # SET_TONE, SYSTEM_SETTING, SET_ALERT usually don't have category
            if act in ['SET_TONE', 'SYSTEM_SETTING', 'SET_ALERT', 'SET_GOAL', 'ADD_GOAL'] and cat != 'None':
                anomalies.append(f'Dòng {i}: Action {act} thường không có category, nhưng lại gán ("{cat}").')
                
        elif intent == 'Chitchat':
            if cat != 'None' or rec != 'None' or act != 'None':
                anomalies.append(f'Dòng {i}: Chitchat nhưng chứa nhãn phụ (cat="{cat}", rec="{rec}", act="{act}").')

with open(r'd:\Luan-Van\Project\expense-ocr-nlu\text_nlu\datasets\jsonl_report.txt', 'w', encoding='utf-8') as f:
    if anomalies:
        for a in anomalies:
            f.write(a + '\n')
    else:
        f.write('Mọi dữ liệu trong JSONL đều HOÀN HẢO. Không có điểm bất thường nào.')
