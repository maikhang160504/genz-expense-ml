import os
import re
import pandas as pd
import psycopg2
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "text_nlu" / "datasets"
RECORD_CSV = DATASETS_DIR / "intent_record.csv"
ACTION_CSV = DATASETS_DIR / "intent_action.csv"
CHITCHAT_CSV = DATASETS_DIR / "intent_chitchat.csv"
BACKEND_ENV = ROOT.parent / "app" / "backend" / ".env"

def get_db_url():
    if not BACKEND_ENV.exists():
        return None
    with open(BACKEND_ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

def load_dfs():
    dfs = {}
    for path, key in [(RECORD_CSV, 'Record'), (ACTION_CSV, 'Action'), (CHITCHAT_CSV, 'Chit-chat')]:
        if path.exists():
            dfs[key] = pd.read_csv(path, encoding="utf-8-sig")
            # ensure columns exist
            if 'category_code' not in dfs[key].columns:
                dfs[key]['category_code'] = ''
        else:
            dfs[key] = pd.DataFrame(columns=['text', 'intent', 'category_code'])
    return dfs

def save_dfs(dfs):
    for path, key in [(RECORD_CSV, 'Record'), (ACTION_CSV, 'Action'), (CHITCHAT_CSV, 'Chit-chat')]:
        dfs[key].to_csv(path, index=False, encoding="utf-8-sig")

def main():
    print(">>> [export_dataset] Bắt đầu đồng bộ dữ liệu từ CSDL PostgreSQL (nlu_logs)...")
    db_url = get_db_url()
    if not db_url:
        print(">>> [export_dataset] Không tìm thấy DATABASE_URL trong app/backend/.env. Bỏ qua đồng bộ DB.")
        return

    dfs = load_dfs()
    initial_total = sum(len(df) for df in dfs.values())
    print(f"Tổng số bản ghi ban đầu: {initial_total}")
    
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT text_input, payload FROM nlu_logs WHERE log_type = 'dislike'")
        rows = cur.fetchall()
        
        updates_count = 0
        new_count = 0
        
        for text_input, payload in rows:
            if not payload or not isinstance(payload, dict):
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except:
                        continue

            if not payload.get("intent_disliked"):
                continue
                
            corrected_intent = payload.get("corrected_intent")
            corrected_category = payload.get("corrected_category")
            
            # Lọc danh mục cá nhân hóa (chỉ nhận danh mục hệ thống bắt đầu bằng cat_)
            if corrected_category and not corrected_category.startswith('cat_'):
                corrected_category = None
                
            if not corrected_intent and not corrected_category:
                continue
                
            text_norm = text_input.strip().lower()
            
            # Xác định target intent (mặc định là Record nếu có category)
            target_intent = corrected_intent or 'Record'
            if target_intent not in dfs:
                target_intent = 'Record'

            found = False
            # Tìm trong cả 3 tập xem có text này chưa
            for key, df in dfs.items():
                mask = df['text'].str.strip().str.lower() == text_norm
                if mask.any():
                    found = True
                    # Nếu intent thay đổi, ta cần di chuyển row sang tập mới
                    if key != target_intent:
                        row_data = df.loc[mask].iloc[0].copy()
                        row_data['intent'] = target_intent
                        if corrected_category:
                            row_data['category_code'] = corrected_category
                        # Xóa ở tập cũ
                        dfs[key] = df[~mask]
                        # Thêm vào tập mới
                        dfs[target_intent] = pd.concat([dfs[target_intent], pd.DataFrame([row_data])], ignore_index=True)
                    else:
                        # Cập nhật tại chỗ
                        if corrected_intent:
                            df.loc[mask, 'intent'] = corrected_intent
                        if corrected_category:
                            df.loc[mask, 'category_code'] = corrected_category
                    updates_count += 1
                    break
            
            if not found:
                new_row = {
                    'text': text_input,
                    'intent': target_intent,
                    'category_code': corrected_category or ''
                }
                dfs[target_intent] = pd.concat([dfs[target_intent], pd.DataFrame([new_row])], ignore_index=True)
                new_count += 1

        cur.close()
        conn.close()
        
        print(f">>> [export_dataset] Đã cập nhật {updates_count} bản ghi cũ, thêm mới {new_count} bản ghi từ feedback của người dùng.")
        
        save_dfs(dfs)
        print(">>> [export_dataset] Đã lưu dataset cập nhật vào các file CSV thành công!")
        
    except Exception as e:
        print(f">>> [export_dataset] Lỗi khi kết nối hoặc đồng bộ DB: {e}")

if __name__ == "__main__":
    main()
