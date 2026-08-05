import os
import re
import pandas as pd
import psycopg2
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = ROOT / "text_nlu" / "datasets" / "intent_action.csv"
BACKEND_ENV = ROOT.parent / "app" / "backend" / ".env"

def get_db_url():
    if not BACKEND_ENV.exists():
        return None
    with open(BACKEND_ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

def main():
    print(">>> [export_dataset] Bắt đầu đồng bộ dữ liệu từ CSDL PostgreSQL (nlu_logs)...")
    if not DATASET_CSV.exists():
        print(f"Không tìm thấy file gốc {DATASET_CSV}. Hủy export.")
        return

    db_url = get_db_url()
    if not db_url:
        print(">>> [export_dataset] Không tìm thấy DATABASE_URL trong app/backend/.env. Bỏ qua đồng bộ DB.")
        return

    # Load file CSV gốc
    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    print(f"Tổng số bản ghi ban đầu: {len(df)}")
    
    # Kết nối DB và lấy danh sách dislikes
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        # Lấy các sample bị report sai
        cur.execute("SELECT text_input, payload FROM nlu_logs WHERE log_type = 'dislike'")
        rows = cur.fetchall()
        
        updates_count = 0
        new_count = 0
        
        for text_input, payload in rows:
            if not payload or not isinstance(payload, dict):
                # parse if string
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except:
                        continue

            if not payload.get("intent_disliked"):
                continue
                
            corrected_intent = payload.get("corrected_intent")
            corrected_category = payload.get("corrected_category")
            
            if not corrected_intent and not corrected_category:
                continue
                
            # Chuẩn hóa text
            text_norm = text_input.strip().lower()
            
            # Tìm xem đã có trong dataframe chưa (bằng text)
            mask = df['text'].str.strip().str.lower() == text_norm
            if mask.any():
                # Update
                if corrected_intent:
                    df.loc[mask, 'intent'] = corrected_intent
                if corrected_category:
                    df.loc[mask, 'category_code'] = corrected_category
                updates_count += 1
            else:
                # Add new row
                new_row = {
                    'text': text_input,
                    'intent': corrected_intent or 'Record',
                    'category_code': corrected_category or ''
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                new_count += 1

        cur.close()
        conn.close()
        
        print(f">>> [export_dataset] Đã cập nhật {updates_count} bản ghi cũ, thêm mới {new_count} bản ghi từ feedback của người dùng.")
        
        # Ghi đè file CSV hoặc lưu ra file train_dataset.csv
        # Để an toàn cho gốc, ta có thể ghi đè vì dataset là để training
        df.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")
        print(">>> [export_dataset] Đã lưu dataset cập nhật thành công!")
        
    except Exception as e:
        print(f">>> [export_dataset] Lỗi khi kết nối hoặc đồng bộ DB: {e}")

if __name__ == "__main__":
    main()
