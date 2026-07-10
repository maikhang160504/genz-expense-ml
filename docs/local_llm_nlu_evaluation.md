# Đánh giá: Fine-tune LLM local cho NLU nhận dạng (Mimo)

> Tài liệu đọc trước khi quyết định có chuyển pipeline NLU sang LLM local hay không.  
> Ngữ cảnh: dataset `intent_action.csv` ~30k dòng (gộp thủ công), pipeline hiện tại TF-IDF + spaCy NER, retrain qua Kaggle/WebAdmin.

---

## 1. Kết luận nhanh

| Câu hỏi | Trả lời |
|---------|---------|
| **LLM local có ổn hơn pipeline hiện tại không?** | **Chưa.** Với metrics hiện tại, **không đáng thay production**. |
| **Có nên thử không?** | **Có**, nhưng nhánh thử nghiệm song song — không bỏ TF-IDF/Kaggle. |
| **Bước trung gian hợp lý hơn full LLM?** | **PhoBERT encoder** (đã có trong repo: `retrain_encoders.py`, `NLU_USE_ENCODER=1`). |
| **Khi nào LLM local mới đáng đầu tư?** | Khi cần **một model** làm cả NLU + chitchat, hoặc TF-IDF/PhoBERT **plateau** trên regression thực tế dù đã sửa data. |

**Khuyến nghị thực tế:**

```
Giữ production: TF-IDF + action slots + spaCy NER
Thử A/B:        PhoBERT encoder (đã có code)
Pilot (sau):    QLoRA 3B–7B tiếng Việt → JSON NLU unified
Không làm ngay: Thay toàn bộ pipeline bằng LLM generative
```

---

## 2. Pipeline hiện tại (baseline)

### 2.1 Kiến trúc

```
User text
    │
    ├─ intent_model (TF-IDF + LogisticRegression)
    │      → Record | Action | Chitchat
    │
    ├─ [Record] record_type + category + NER (spaCy) + amount regex
    ├─ [Action] action_type_model + action_slots_model (11 field heads)
    └─ [Chitchat] LLM API (Gemini) — KHÔNG train local
```

### 2.2 Metrics sau retrain (dataset merged ~30k action)

| Model | Weighted F1 / Accuracy | Ghi chú |
|-------|------------------------|---------|
| Intent | **99%** | Record / Action / Chitchat |
| Record type | **98%** | Income / Expense |
| Category | **95%** | 12 danh mục |
| Action type | **100%** | 13 loại |
| Action slots (avg) | **95% F1**, 94% acc | 11 field |
| → time_range | ~90% | Nhiều biến thể không dấu / teencode |
| → note | ~65% | Quá nhiều class, data nhiễu |
| NER (record) | Riêng spaCy | PRODUCT, AMOUNT, TIME, … |

**Inference:** vài ms trên CPU, model joblib nhẹ (~vài MB–vài chục MB).

### 2.3 LLM đang dùng ở đâu?

| Vai trò | Công nghệ | Train local? |
|---------|-----------|--------------|
| Sinh dataset (đã dừng) | Gemini API | Không — user gộp CSV thủ công |
| Chitchat / NLG runtime | Gemini API | Không |
| NLU nhận dạng production | TF-IDF + sklearn | Có — Kaggle/local |

→ Câu hỏi của bạn là: **dùng LLM local fine-tune trên CSV để thay phần NLU nhận dạng** — khác với LLM chỉ sinh data hoặc trả lời chitchat.

---

## 3. Ba mức “model ngôn ngữ” — không nhầm lẫn

| Mức | Ví dụ | Output | Repo Mimo |
|-----|-------|--------|-----------|
| **A. Classical** | TF-IDF + LogisticRegression | Label / slot từng head | **Production** |
| **B. Encoder + head** | PhoBERT embed + classifier | Label / slot từng head | **Có sẵn, experimental** |
| **C. Generative LLM** | Qwen2.5-7B, VinLlama, … QLoRA | JSON text (intent, slots, …) | **Chưa có** |

Fine-tune LLM local trong câu hỏi của bạn = **mức C**.  
Repo đã có **mức B** (`train_intent_encoder.py`, `encoder_runtime.py`) — đây là bước nhảy hợp lý trước khi lên C.

---

## 4. So sánh chi tiết

### 4.1 Độ chính xác (accuracy)

| Tiêu chí | TF-IDF (hiện tại) | PhoBERT encoder | LLM 3B–7B fine-tune |
|----------|-------------------|-----------------|---------------------|
| Intent 3 lớp | 99% — rất cao | Thường tương đương hoặc +0–2% | Có thể +1–3% trên câu lạ |
| Action type 13 lớp | 100% trên test set | Tương đương | Không cần thiết trước mắt |
| Slots có schema cố định | 95% avg; note 65% | Có thể cải thiện note/time_range nhẹ | Cải thiện **nếu** prompt + normalize tốt; **dễ hallucinate value** |
| NER span (PRODUCT, …) | spaCy chuyên biệt | Cần model riêng hoặc token classification | LLM extract JSON — cần validate chặt |
| Teencode / không dấu | TF-IDF + tokenize PyVi ổn | **Tốt hơn** nhờ semantic | **Tốt hơn** nếu pretrain Việt đủ |

**Nhận xét:** Với test set hiện tại, classical đã gần trần. LLM local **không tự động** vượt trội — lợi thế chủ yếu ở **câu OOD** (out-of-distribution), lỗi chính tả nặng, câu ghép nhiều ý.

### 4.2 Độ trễ & tài nguyên

| | TF-IDF | PhoBERT-base | LLM 7B QLoRA inference |
|--|--------|--------------|-------------------------|
| Latency / câu | **1–10 ms** (CPU) | 30–150 ms (GPU) / 200–500 ms (CPU) | **0.5–3 s** (GPU 8GB+) |
| RAM/VRAM inference | < 500 MB | ~1–2 GB | **4–8 GB+** |
| Fine-tune | CPU vài phút | GPU 1–2 giờ (embed cache) | GPU 4–12 giờ (QLoRA) |
| Deploy ai-service | joblib load | torch + transformers | vLLM / llama.cpp / Ollama |

App chat mobile cần phản hồi nhanh → TF-IDF vẫn phù hợp cho **hot path**.

### 4.3 Độ tin cậy output có cấu trúc

NLU Mimo cần:

- `action_type` ∈ 13 giá trị cố định
- `value` = số VND (integer)
- `category_code` ∈ 12 mã
- `enabled` = `"true"` | `"false"`

| | TF-IDF / sklearn | LLM generative |
|--|------------------|----------------|
| Schema cứng | ✅ Mỗ head chỉ predict class hợp lệ | ⚠️ Có thể sinh `"Foodd"`, `"50000k"`, thiếu field |
| Giải pháp | Built-in | **Bắt buộc:** JSON schema + parser + fallback TF-IDF |
| Regression `value` | Ridge regressor riêng | LLM hay format sai đơn vị |

→ LLM local **cần lớp validation** giống backend hiện tại — không thể “tin 100%” output.

### 4.4 Chi phí vận hành & retrain

| | Pipeline Kaggle hiện tại | LLM local |
|--|--------------------------|-----------|
| Retrain từ WebAdmin | ✅ Đã tích hợp | Cần pipeline mới (Unsloth/LLaMA-Factory) |
| Sync model về ai-service | joblib zip | GGUF / LoRA adapter / full weights |
| Metrics WebAdmin | ✅ `retrain_all_metrics.json`, action_slots | Cần benchmark script riêng |
| Debug lỗi | Confusion matrix, per-field F1 | Khó trace — prompt + generation randomness |

---

## 5. Vấn đề thực sự cần sửa (rẻ hơn đổi LLM)

Trước khi fine-tune LLM, nên xử lý các điểm yếu **đã đo được**:

### 5.1 Slot `note` — F1 ~65%

- Nguyên nhân thường gặp: quá nhiều class tự do, nhãn không nhất quán, trùng nghĩa.
- **Fix data:** gom note thành taxonomy nhỏ hoặc chuyển sang NER span thay vì classify.
- LLM có thể giúp **sinh lại nhãn** (offline), không nhất thiết phải **inference** bằng LLM.

### 5.2 Slot `time_range` — F1 ~90%

- Nhiều biến thể: `tuan nay` vs `tuần này` vs `tuần này nha`.
- **Fix:** normalize về canonical form trong `label_action_slots.py` / `time_parser.py`.
- TF-IDF sẽ nhảy lên mà không cần LLM.

### 5.3 Intent Action vs Record

- Regex tiền + intent 99% — LLM không mang lại ROI rõ ràng ở đây.

---

## 6. Khi nào LLM local **đáng** thử?

Chọn **một hoặc nhiều** điều kiện sau:

1. **Unified model:** Một model 3B–7B vừa classify intent/action vừa trả lời chitchat (giảm phụ thuộc Gemini API).
2. **Regression thực tế:** Bộ test 200–500 câu thực tế (không có trong train) mà TF-IDF + PhoBERT đều < 85%.
3. **Multi-intent / câu dài:** Ví dụ *"tuần này ăn uống bao nhiêu rồi, set limit 2tr luôn"* — generative dễ hơn pipeline tuần tự.
4. **Có GPU cố định** trên server ai-service (≥ 8GB VRAM cho 7B quantize).
5. **Team sẵn sàng** maintain thêm stack (transformers, vLLM, eval JSON).

---

## 7. Lộ trình đề xuất (nếu muốn thử)

### Phase 0 — Giữ nguyên (0–2 tuần)

- Production: TF-IDF + Kaggle retrain.
- Sửa data: normalize `time_range`, audit `note`.
- Chạy full `retrain_all.py` trên dataset 30k đã gộp.

### Phase 1 — A/B PhoBERT (1 tuần, ít rủi ro)

Repo đã có:

```powershell
cd expense-ocr-nlu
.\.venv\Scripts\Activate.ps1
python text_nlu/train/retrain_encoders.py
python text_nlu/tools/compare_tfidf_encoder.py --samples "set limit ăn uống 2tr tuần này"
```

Bật thử trên ai-service:

```env
NLU_USE_ENCODER=1
```

So sánh trên cùng bộ regression → quyết định có lên Phase 2 không.

### Phase 2 — Pilot LLM JSON NLU (2–4 tuần)

**Mục tiêu:** Một model generative output JSON, **không thay production**.

| Hạng mục | Gợi ý |
|----------|-------|
| Base model | `Qwen2.5-3B-Instruct`, `vinallama-7b`, hoặc `SeaLLM-7B` |
| Method | QLoRA (4-bit), 1–3 epoch |
| Format train | Alpaca / ShareGPT: `instruction + input (text) → output (JSON)` |
| Data | Merge CSV hiện có → ~100k samples (record + action + chitchat) |
| Tool | LLaMA-Factory, Unsloth, hoặc axolotl |
| Eval | Script so `predicted JSON` vs gold; per-field F1 giống `action_slots_metrics.json` |
| Deploy thử | Ollama / llama.cpp sidecar; ai-service gọi HTTP nội bộ |

**Output JSON mẫu (action):**

```json
{
  "intent": "Action",
  "action_type": "SET_LIMIT",
  "slots": {
    "verb": "SET",
    "category_code": "Food",
    "value": 2000000,
    "time_range": "tuần này"
  }
}
```

**Luồng hybrid an toàn:**

```
User text → TF-IDF (nhanh, chính)
         → nếu confidence < 0.75 → LLM JSON (fallback)
         → validate schema → backend action.service
```

### Phase 3 — Chỉ khi Phase 2 thắng regression

- Thay hot path hoặc gộp chitchat vào cùng model.
- Tích hợp metrics vào WebAdmin (`NluOpsPage`) tương tự `action_slots`.

---

## 8. Dataset cho fine-tune LLM — tái sử dụng CSV hiện có

Bạn **không cần** sinh dataset mới bằng API. Chuyển đổi từ file có sẵn:

| File | Dùng cho |
|------|----------|
| `intent_record.csv` | intent=Record + category + amount (từ text/NER) |
| `intent_action.csv` | intent=Action + action_type + 11 slots |
| `intent_chitchat.csv` | intent=Chitchat (+ optional response nếu có) |

Script chuyển đổi (concept):

```python
# text → JSON string làm assistant content
{"intent":"Action","action_type":"SUGGEST_BUDGET","slots":{"time_range":"tháng này"}}
```

**Lưu ý:**

- Train/val/test split **theo text hash** — tránh leak duplicate gần giống.
- Oversample Chitchat giống `train_intent_model.py`.
- Augment teencode **có kiểm soát** — không chỉ LLM generate thêm (đã dừng API gen).

---

## 9. Rủi ro & cách giảm

| Rủi ro | Giảm thiểu |
|--------|------------|
| Hallucinate số tiền | Validate bằng regex + so với NER/amount extractor |
| Latency chat | Chỉ fallback LLM; hoặc model 3B quantize |
| Model drift sau fine-tune | Giữ golden test set 500 câu; CI regression |
| VRAM không đủ | 3B Q4_K_M; hoặc chỉ inference trên Kaggle/Colab trước |
| Mất pipeline Kaggle | LLM train **song song**, không sửa `retrain_all.py` cho đến khi pass eval |

---

## 10. Bảng quyết định

| Bạn muốn… | Làm gì |
|-----------|--------|
| Retrain ổn định, metrics WebAdmin | **Tiếp tục Kaggle + TF-IDF** ✅ |
| Cải thiện note / time_range | **Sửa data + normalize** trước ✅ |
| Thử model “thông minh” hơn, ít code | **PhoBERT encoder A/B** ✅ |
| Bỏ Gemini cho chitchat + NLU một model | **Pilot LLM QLoRA Phase 2** ⚠️ |
| Thay hết TF-IDF ngay vì “LLM trendy” | **Không khuyến nghị** ❌ |

---

## 11. Tóm tắt một câu

**Fine-tune LLM local trên dataset CSV hiện có là khả thi và hợp lý cho pilot / fallback / gộp chitchat — nhưng với F1 95–99% trên pipeline classical, nó chưa “ổn hơn” đủ để thay production; hãy sửa slot yếu (note, time_range), chạy PhoBERT A/B, rồi mới pilot QLoRA JSON nếu regression thực tế vẫn thiếu.**

---

## Phụ lục — File liên quan trong repo

| File | Vai trò |
|------|---------|
| `text_nlu/train/retrain_all.py` | Pipeline production |
| `text_nlu/train/train_action_slots.py` | Slot heads + metrics |
| `text_nlu/train/retrain_encoders.py` | PhoBERT experimental |
| `src/nlu/models.py` | Load TF-IDF vs encoder |
| `src/nlu/encoder_runtime.py` | Inference PhoBERT |
| `text_nlu/tools/compare_tfidf_encoder.py` | So sánh nhanh |
| `app/frontend/web-admin/src/pages/NluOpsPage.jsx` | Hiển thị metrics |
| `text_nlu/datasets/dataset_generation_prompts.md` | Prompt sinh data (LLM API, đã dừng) |

---

*Tài liệu v1 — 2025-06-25. Cập nhật sau khi có kết quả `compare_tfidf_encoder` trên bộ regression thực tế.*
