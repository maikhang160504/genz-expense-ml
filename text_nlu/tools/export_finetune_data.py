import json
import os
import psycopg2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = ROOT / "src" / "prompts" / "prompts.json"
OUTPUT_PATH = ROOT / "text_nlu" / "datasets" / "dataset_finetune.jsonl"
OUTPUT_STORAGE_PATH = Path("/storage/exported/vistral_finetune_incremental.jsonl")
BACKEND_ENV = ROOT.parent / "app" / "backend" / ".env"

def get_db_url():
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    if not BACKEND_ENV.exists():
        return None
    with open(BACKEND_ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

def export_finetune_data():
    print(">>> Bắt đầu tạo dữ liệu fine-tune từ CSDL PostgreSQL (nlu_logs)")
    
    if not PROMPTS_PATH.exists():
        print("Không tìm thấy file prompts.json")
        return
        
    db_url = get_db_url()
    if not db_url:
        print(">>> Lỗi: Không tìm thấy DATABASE_URL trong app/backend/.env.")
        return

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        
    instruction = prompts.get("llm_unified_prompt", {}).get("system", "Phân tích câu thoại và trả về JSON.")
    
    results = []
    
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        # Lấy các log chat thực tế
        cur.execute("SELECT text_input, intent, category, payload FROM nlu_logs WHERE log_type = 'chat'")
        rows = cur.fetchall()
        
        seen_texts = set()
        for text_input, intent, category, payload in rows:
            if not text_input or not str(text_input).strip():
                continue
                
            text_norm = str(text_input).strip().lower()
            if text_norm in seen_texts:
                continue
            seen_texts.add(text_norm)
            
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    payload = {}
                    
            if not payload:
                continue
                
            # Parse thông tin ngữ cảnh nếu có trong payload
            context_meta = payload.get("contextMeta", {})
            input_text = f"Ngữ cảnh hệ thống (CONTEXT_META): {json.dumps(context_meta, ensure_ascii=False)}\nCâu thoại của người dùng: {text_input}"
            
            # Khôi phục Output JSON chuẩn từ dữ liệu RAG đã sinh
            # AI trả về thường nằm ở `payload` (trong Node.js lưu vào)
            output_obj = {
                "intent": intent or payload.get("intent", "Record"),
                "action_type": payload.get("action_type"),
                "slots": payload.get("slots", {}),
                "suggested_actions": payload.get("suggested_actions"),
                "emotion": payload.get("emotion", "neutral"),
                "response": payload.get("response", "Đã hiểu yêu cầu.")
            }
            
            # Ghi category vào slots nếu category DB không rỗng (RAG)
            if category and not output_obj["slots"].get("category"):
                output_obj["slots"]["category"] = category
            
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(output_obj, ensure_ascii=False)
            })

        cur.close()
        conn.close()
        
    except Exception as e:
        print(f">>> Lỗi kết nối CSDL: {e}")
        return
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Sao chép vào /storage để dùng cho huấn luyện Modal nếu thư mục gốc /storage tồn tại
    if Path("/storage").exists() or OUTPUT_STORAGE_PATH.parent.exists():
        import shutil
        OUTPUT_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, OUTPUT_STORAGE_PATH)
        print(f">>> Đã copy dữ liệu sang Modal Storage: {OUTPUT_STORAGE_PATH}")
            
    print(f">>> Đã xuất {len(results)} mẫu thực tế từ DB ra {OUTPUT_PATH}")
    return str(OUTPUT_PATH)

if __name__ == "__main__":
    export_finetune_data()
