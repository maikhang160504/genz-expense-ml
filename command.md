# Lệnh train & demo — NLU text (phase đóng · chuẩn bị OCR)

Thư mục: `D:\Luan-Van\Train`

```powershell
Set-Location D:\Luan-Van\Train
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING='utf-8'
```

**Chitchat:** intent NLU → **Gemini/LLM** (không train PhoBERT sentiment).  
**Backends:** intent/action `encoder` · category/record_type `tfidf` (+ encoder nếu có file).

---

## 1. Demo nhanh (không train)

### Smoke + verify cố định

```powershell
python text_nlu\tools\smoke_intent_samples.py
python text_nlu\tools\verify_task01.py
python text_nlu\tools\verify_task03.py
python text_nlu\tools\verify_task05.py
python text_nlu\tools\demo_nlu_random20.py
```

### Demo chat tương tác (NLU + Gemini)

Cần `.env`: `gemini_API`, `GEMINI_MODEL=gemini-2.5-flash`

```powershell
$env:RUN_LLM='1'
$env:RUN_LLM_CHITCHAT='1'
$env:LLM_MODE='gemini' 
python -m src.cli.demo_inference
```

### Demo câu chi tiêu sinh viên (Gemini sinh → NLU)

```powershell
python text_nlu\tools\gemini_demo_student_expenses.py --count 40 --nlu
# Gemini 503: retry tự động + đổi model; hoặc không API:
python text_nlu\tools\gemini_demo_student_expenses.py --count 40 --nlu --local
# Tuỳ chọn: $env:GEMINI_MODEL='gemini-2.0-flash'
```

Đầu ra: `text_nlu/tools/demo_student_expenses.json`, `demo_student_expenses.txt`

### API FastAPI

```powershell
$env:RUN_LLM='1'
$env:RUN_LLM_CHITCHAT='1'
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

`POST /infer` body mẫu:

```json
{
  "text": "ăn phở 45k",
  "profile": {},
  "run_llm": true
}
```

---

## 2. Chuẩn bị dữ liệu (~15k biên mua/bán, cafe…)

Cần `.env` nếu dùng Gemini: `gemini_API`

```powershell
# Sinh ~15k từ mô tả dữ liệu.md (local, không API)
python text_nlu\datasets\generate_dataset_15k.py local --target 15000

# Tuỳ chọn thêm bằng Gemini (80 batch × 100 dòng)
python text_nlu\datasets\generate_dataset_15k.py gemini --batches 80 --rows 100

# Hoặc local + Gemini + sửa nhãn cũ
python text_nlu\datasets\generate_dataset_15k.py all --target 15000

# Chỉ sửa nhãn CSV hiện có
python text_nlu\datasets\fix_disambiguation_labels.py

# Cải thiện chất lượng dataset
python text_nlu\datasets\improve_datasets.py
```

---

## 3. Train model (theo nhu cầu)

```powershell
# TF-IDF
python text_nlu\train\train_category_model.py
python text_nlu\train\train_record_type_model.py

# Encoder (intent + action) — không gồm sentiment
python text_nlu\train\retrain_encoders.py

# Tuỳ chọn
python text_nlu\train\train_record_type_encoder.py
python text_nlu\train\train_category_encoder.py
```

**Không còn dùng:** `train_chitchat_sentiment_hf.py` (đã xóa — Chitchat = LLM).

---

## 4. Train NER (tuỳ chọn)

```powershell
python text_nlu\train\ner_prepare.py
python text_nlu\train\train_ner_only.py
```

---

## 5. Biến môi trường (.env)

| Biến | Mục đích |
|------|----------|
| `gemini_API` | Gemini NLG + demo sinh viên |
| `GEMINI_MODEL` | Mặc định `gemini-3.5-flash` |
| `RUN_LLM` | `1` = gọi LLM mọi intent |
| `RUN_LLM_CHITCHAT` | `1` = Chitchat luôn gọi LLM (mặc định) |
| `LLM_MODE` | `gemini` \| `both` |
| `USE_MOCK_CONTEXT` | `1` mock / `0` profile cố định (CLI) |

---

## 6. Train lại sau khi thêm data (thay pipeline hints)

Biên **mua/bán**, **đi cafe vs mua cafe**, **gạo/sạc** học từ model — không còn rule `*_hints.py`.

```powershell
python text_nlu\train\train_category_model.py
python text_nlu\train\train_record_type_model.py
python text_nlu\train\retrain_encoders.py
python text_nlu\tools\smoke_intent_samples.py
```

---

## 7. Model cũ (đã archive)

Sentiment PhoBERT không dùng — xem `archive/deprecated_sentiment/README.md`.

---

## 8. Triển khai API Server Hợp Nhất (FastAPI)

Kể từ phiên bản hợp nhất, server chạy trực tiếp từ `src.api.app:app` đã bao gồm toàn bộ tính năng **NLU (Text)**, **OCR (Image)** và **Gán nhãn & Retrain**.

```powershell
# Chạy DEV server
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Các Endpoint chính:
*   `POST /api/v1/nlu/infer`: Nhận diện ý định, thực thể, và sinh câu trả lời Mimo (NLU text).
*   `POST /api/v1/ocr/image`: Đọc ảnh hóa đơn bằng PaddleOCR + VietOCR + PICK KIE và trả về tổng tiền, sản phẩm, category.
*   `POST /api/v1/bill-retrain/predict`: Dự đoán KIE trên 1 ảnh và trả về dữ liệu OCR kèm nhãn tự động chuẩn hóa bởi LLM.
*   `POST /api/v1/bill-retrain/train`: Quản lý trigger retrain mô hình PICK KIE dựa trên các nhãn đã xác thực.

---

## 9. Huấn luyện lại và Gán nhãn hóa đơn (bill_ocr)

Toàn bộ luồng huấn luyện PICK KIE, gán nhãn tự động bằng LLM và sửa nhãn thủ công (CLI) nằm trong thư mục `bill_ocr/`.
*   Hướng dẫn chạy train Docker GPU và kiểm thử local: **[`bill_ocr/DOCKER_INSTRUCTIONS.md`](bill_ocr/DOCKER_INSTRUCTIONS.md)**.
*   Hướng dẫn triển khai API sản xuất & chạy train Serverless GPU trên **Modal**: **[`docs/MODAL_DEPLOYMENT.md`](docs/MODAL_DEPLOYMENT.md)**.
