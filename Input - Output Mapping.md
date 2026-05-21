# Input — Output Mapping (đồng bộ với pipeline hiện tại)

Tài liệu mô tả **luồng thật trong repo** (NLU → nhánh intent → NLG/LLM). Bộ phân loại intent hiện có **ba nhãn**: `Record`, `Action`, `Chitchat`.  
Các ví dụ nhóm **Thống kê / Analytics** trong tài liệu sản phẩm được quy ước: *trong stack hiện tại, các câu kiểu “tháng này tiêu bao nhiêu” thường được gán intent **Action** (ví dụ `action_type` Report / Search)*; bước **tổng hợp dữ liệu / SQL** sẽ cắm **sau NLU, trước prompt LLM** khi backend có cơ sở dữ liệu — chưa nằm trong demo CLI.

---

## 0. Sơ đồ luồng tổng quát

1. **NLU**: `run_nlu` → intent (`Record` | `Action` | `Chitchat`), thực thể/số tiền (NER + TF-IDF phụ), `action_type` nếu là Action, `sentiment` nếu là Chitchat.  
2. **Context (ngữ cảnh prompt)**:
   - **Production / API** (`POST /infer`): `build_context_metadata(nlu_result, profile)` — module `src/nlg/context_meta.py` (import tương thích qua `src/nlg/metadata.py`). Chỉ gửi kèm prompt khi có `profile` (hoặc sau này điều kiện nghiệp vụ).
   - **Demo**: mặc định `USE_MOCK_CONTEXT=1` → `build_mock_context_metadata(...)` — **dữ liệu ngẫu nhiên**, minh họa chỗ “đính kèm context” vào prompt.  
3. **NLG**: `build_nlg_prompt` lấy template theo `intent` và `emotion` (`src/prompts/prompts.json`).  
4. **LLM** (tùy chọn): fusion text = JSON NLU + JSON context → model trả `story` + `status` (JSON).

---

## 1. Nhóm Ghi chép (**Record**)

| Mục tiêu | Lưu nhanh chi thu, phản hồi có cảm xúc (story). |
| Input (ví dụ) | Logic backend (đang / sắp có) | Output app |
|----------------|--------------------------------|------------|
| “Ăn bát phở 50k” | NLU: intent Record, amount, item/category… Context: so khớp với `budget_remain` / ngưỡng → cờ cảnh báo nếu lố | Story JSON: `story`, `status` |

**Điều chỉnh flow (đã phản ánh trong code `context_meta`):**  
So sánh `amount` với `amount_threshold`, `budget_remain` / `budget_total` (còn &lt; 20% tổng hoặc chi vượt dư **hoặc** vượt ngưỡng) và tần suất → `is_triggered`, `type`.

---

## 2. Nhóm Hành động (**Action**)

| Mục tiêu | Thực hiện lệnh (ngân sách, lịch sử, tone…). |
| Input (ví dụ) | Logic | Output |
|----------------|-------|--------|
| “Đặt lại giới hạn chi tiêu” | `action_type` (Reset / Setting / …), tham số | **Action Ack** ngắn — không “chuyện giao dịch” dài |

**Flow:** Prompt **Action** dùng `action_user` + `action_response_rules`: một câu xác nhận trong trường `story` của JSON, không bịa giao dịch mới. Demo tắt LLM vẫn có `action_ack` cục bộ.

---

## 3. Thống kê (**Analytics** trong sản phẩm → **Action** trong model hiện tại)

| Mục tiêu | Con số + gợi ý. |
| Input | Sau NLU: truy vấn / tổng hợp (SQL hoặc service) → **đưa kết quả vào prompt** làm facts. |
| Output | “Analytics summary” do LLM tóm tắt từ dữ liệu aggregation. |

**Lưu ý:** Khi đã có SQL, nên map kết quả vào một block JSON cố định trong user prompt (tương tự `context_metadata` + fusion), tránh model tự bịa số.

---

## 4. Nhóm Chitchat (**Chitchat**)

| Mục tiêu | Tán gẫu + **cầu nối** về ghi chi tiêu. |
| Không gọi DB bắt buộc trong demo. |
| Prompt: `chitchat_user` + `chitchat_response_rules` — trả lời vui, sau đó gợi ý ghi chi. |

---

## 5. Out-of-scope / độ tin cậy thấp (hướng **sau này**)

- Intent threshold (confidence &lt; 0.5 → fallback) **chưa** wired trong demo — có thể thêm trước NLG.  
- Ví dụ OOS “Thời tiết hôm nay?” → hiện có thể rơi **Chitchat** hoặc **Action** sai; cần tập dữ liệu + routing riêng.

---

## 6. File liên quan trong repo

| Thành phần | Đường dẫn |
|-------------|-----------|
| NLU | `src/nlu/pipeline.py` |
| Context (profile + mock) | `src/nlg/context_meta.py` (shim: `src/nlg/metadata.py`) |
| Prompt builder | `src/nlg/prompt.py` |
| Nội dung prompt JSON | `src/prompts/prompts.json` |
| Demo CLI | `src/cli/demo_inference.py` |
| API | `src/api/app.py` (profile thật → `build_context_metadata`) |
| Ví dụ payload Gemini | `format_request_reponse.md` |

---

## 7. Biến môi trường demo

| Biến | Ý nghĩa |
|-------|---------|
| `USE_MOCK_CONTEXT=1` (mặc định) | Context đính kèm prompt = **mock ngẫu nhiên** |
| `USE_MOCK_CONTEXT=0` | Context từ **profile** cố định trong `demo_inference` |
| `RUN_LLM=1` | Gọi Gemini / Groq nếu có key trong `.env` |
