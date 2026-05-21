# TASK — NLU text (đóng phase) · OCR tiếp theo

**Trạng thái:** `[ ]` chưa · `[~]` một phần · `[x]` xong · `[—]` đóng / không làm

**Lệnh train & demo:** [`command.md`](command.md) · log train: [`REQUEST.MD`](REQUEST.MD) / [`OUTPUT_TASK.MD`](OUTPUT_TASK.MD)

**Kiến trúc chitchat:** Intent NLU → **Gemini/LLM** (tone + story + CTA). Không train PhoBERT sentiment.

**Backends:** intent/action `encoder` · category/record_type `tfidf` (+ encoder nếu có file).

---

## Tóm tắt đã hoàn thành

| Hạng mục | Trạng thái |
|----------|------------|
| Dataset record ~32k+, chitchat ~2.2k, action ~4.6k | [x] |
| TASK-01 income ngắn + `record_type_hints` | [x] |
| TASK-02 smoke/demo 20/20 intent | [x] |
| TASK-03 Shopping + TASK-11 Essentials boundary | [x] data + train + verify |
| TASK-05 REPORT_GENERAL | [x] data + train + verify |
| TASK-12 expense «cho» hard-negative | [x] data + train |
| TASK-15 regression smoke 20/20 | [x] |
| Family bonus (`mẹ cho` → Bonus) | [x] |
| Data biên mua/bán + cafe (~15k, `generate_dataset_15k.py`) | [x] local |
| Pipeline hints → model (đã xóa `*_hints.py`) | [x] cần train lại |
| API `/infer` gọi Gemini như CLI (`llm_runner`) | [x] |
| `command.md` + archive sentiment cũ | [x] |
| TASK-04 / TASK-14 PhoBERT sentiment | [—] thay bằng LLM |
| TASK-08–10 app (popup, học user, NER PRODUCT) | [ ] sau backend |
| TASK-OCR PaddleOCR + VietOCR | [ ] phase mới |

---

## P0 — Production fixes

### TASK-01 — Record type income ngắn — **[x]**

- [x] `append_income_short_rows.py`, `train_record_type_model.py`, `record_type_hints.py`
- [x] `verify_task01.py` 5/5

### TASK-02 — Xác nhận pipeline — **[x]**

- [x] `smoke_intent_samples.py`, `demo_nlu_random20.py`, `gemini_demo_student_expenses.py` 35/35
- [x] Kết quả: `OUTPUT_TASK.MD`

---

## P1 — Dữ liệu & model

### TASK-03 — Category Shopping — **[x]**

- [x] Data + `train_category_model.py` + `category_hints.py`
- [x] `verify_task03.py` 7/7

### TASK-04 — Sentiment chitchat — **[—] ĐÓNG**

Thay bằng **LLM** (`load_chitchat_sentiment_model()` → `backend: llm`). Model cũ trong `archive/deprecated_sentiment/`.

### TASK-05 — Action REPORT_GENERAL — **[x]**

- [x] `append_report_general_rows.py`, `action_intent_hints.py`
- [x] `verify_task05.py` 7/7

### TASK-11 — Essentials boundary — **[x]**

- [x] `append_essentials_boundary_rows.py` · gộp trong train category TASK-03

### TASK-12 — Income/expense «cho» — **[x]**

- [x] Hard-negative + income extra · `verify_task01.py`

### TASK-13 — Record type encoder — **[~] TÙY CHỌN**

- [x] `train_record_type_encoder.py`, pipeline ưu tiên encoder nếu có file
- [ ] Train và so sánh với TF-IDF khi cần (`verify_task13.py`)

### TASK-14 — Sentiment Positive ngắn — **[—] ĐÓNG** (cùng TASK-04)

### TASK-15 — Regression smoke — **[x]**

- [x] 20/20 intent + 22/22 extra fields sau train đầy đủ

### TASK-16 — Category encoder — **[~] TÙY CHỌN**

- [x] `train_category_encoder.py` · `verify_task16.py`
- [ ] Train nếu F1 tốt hơn TF-IDF

---

## P2 — Vận hành & API

### TASK-06 — Docs & lệnh — **[x]**

- [x] [`command.md`](command.md) — train, demo CLI/API, env, hints, OCR-next
- [x] [`REQUEST.MD`](REQUEST.MD) — mẫu train cho agent/user

### P1 API — Gemini trong `/infer` — **[x]**

- [x] `src/nlg/llm_runner.py` dùng chung CLI + `src/api/app.py`
- Body: `"run_llm": true` hoặc Chitchat mặc định `RUN_LLM_CHITCHAT=1`

### TASK-07 — Calibration intent confidence — **[ ] TÙY CHỌN**

- [ ] Script đánh giá `intent_confidence` trên hold-out

### Train sau data 15k — **[ ] BẠN CHẠY**

```powershell
python text_nlu\train\train_category_model.py
python text_nlu\train\train_record_type_model.py
python text_nlu\train\retrain_encoders.py
```

---

## P3 — App (sau backend server)

> Yêu cầu gốc: [`cần sửa.md`](cần%20sửa.md) · **Luồng chốt triển khai:** [`ARCHITECTURE_TRIEN_KHAI.md`](ARCHITECTURE_TRIEN_KHAI.md)

### TASK-08 — Học từ correction user — **[ ]**

- [ ] API corrections + `data/users/{id}/corrections.csv`
- [ ] Merge infer: exact match → user model → global
- [ ] `train_user_models.py` → `text_nlu/models/users/{id}/`

### TASK-09 — Action popup + nhớ xác nhận — **[ ]**

- [ ] DB `user_confirmed_actions`, `action_rejected_log`
- [ ] API confirm/reject (server_module)

### TASK-10 — NER PRODUCT — **[ ]**

---

## TASK-OCR — Phase tiếp theo

**Mục tiêu:** Ảnh hóa đơ / screenshot → text → `run_nlu()` → Record/Action.

**Stack dự kiến:** PaddleOCR + VietOCR (chưa có trong repo).

**Việc làm (khi bắt đầu):**

1. [ ] Module `src/ocr/` hoặc `text_nlu/ocr/` — preprocess ảnh
2. [ ] Pipeline `image → text → run_nlu(text)`
3. [ ] API endpoint `POST /infer-image` (sau khi OCR ổn)
4. [ ] Dataset mẫu hóa đơ VN (tuỳ chọn fine-tune)

**Lệnh chuẩn bị:** xem mục **8** trong [`command.md`](command.md).

---

## Thứ tự đã chạy

```
P0 (TASK-01,02) [x] → P1 data+train (03,05,11,12,15) [x]
→ Chitchat LLM thay sentiment (04,14) [—]
→ P2 command.md + API llm_runner [x]
→ P3 app [ ] sau BE
→ OCR [ ]
```

---

## Ghi chú phần cứng

- **AMD iGPU / CPU** — train PhoBERT encoder trên CPU (đã xong).
- **NVIDIA** — có thể tăng batch/epoch; xem gợi ý trong `REQUEST.MD`.
