# Prompt sinh dataset NLU — Mimo Chat

Ba file CSV dùng train intent + model con. Copy từng prompt, thay `[N]` và tham số cuối prompt trước khi gửi LLM.

Tham chiếu schema: `action_slot_columns.py`, `action.md`.

---

## 1. `intent_action.csv` — Lệnh hệ thống (Action)

```
Bạn là chuyên gia gán nhãn NLU cho app quản lý chi tiêu Mimo (tiếng Việt, Gen Z, teencode, có/không dấu, ngắn gọn).

Nhiệm vụ: sinh file 20k dòng CSV intent_action.csv — mỗi dòng là một câu chat thể hiện LỆNH HỆ THỐNG (không phải ghi nhận chi tiêu, không phải trò chuyện thường).

=== CẤU TRÚC CSV (UTF-8, dấu phẩy, header bắt buộc) ===
text,intent,action_type,verb,category_code,value,goal_name,enabled,theme,verbal_style,time_range,query,note

=== CỘT BẮT BUỘC ===
• category_code: chỉ một trong:
• text (string): Câu người dùng, 4–200 ký tự, tự nhiên, đa dạng, tiếng việt không lẫn tiếng anh(đổi category thành tiếng việt). KHÔNG copy ví dụ mẫu.
• intent (string): LUÔN là Action (chữ A hoa).
• action_type (string): Một trong 13 loại UPPER_SNAKE bên dưới.

=== 13 action_type & CỘT SLOT PHẢI ĐIỀN (cột không dùng để TRỐNG) ===

| action_type      | Cột slot bắt buộc                          |
|------------------|--------------------------------------------|
| SET_LIMIT        | verb, category_code, value                 |
| SET_ALERT        | category_code, enabled                     |
| SET_GOAL         | goal_name, value                           |
| ADD_GOAL         | verb, goal_name, value                     |
| SYSTEM_SETTING   | theme                                      |
| SET_TONE         | verbal_style                               |
| SET_USERNAME     | value (tên hiển thị, có thể chữ)           |
| SET_INCOME       | value                                      |
| SEARCH_RECORD    | query; category_code hoặc value nếu có     |
| REPORT_GENERAL   | time_range; category_code nếu lọc danh mục |
| SUGGEST_BUDGET   | time_range                                 |

=== QUY TẮC GIÁ TRỊ TỪNG CỘT SLOT ===
• verb: chỉ SET | ADD | SUB (SET_LIMIT, ADD_GOAL).
• category_code: chỉ một trong:
  Food, Essentials, Social, Transport, Shopping, Housing, Health, Beauty,
  Education, Entertainment, Investment, Others
• value: số nguyên VND trong CSV (50000, 1000000). Không ghi "k", "tr", "củ" trong cột value — chỉ trong text.
• goal_name: tên mục tiêu tiết kiệm tiếng Việt (vd: "mua laptop mới").
• enabled: chuỗi "true" hoặc "false" (SET_ALERT).
• theme: dark | light (SYSTEM_SETTING).
• verbal_style: funny | gentle | serious | sarcastic | strict (SET_TONE).
• time_range: cụm thời gian tiếng Việt (vd: "tuần này", "tháng trước", "7 ngày qua").
• query: từ khóa tìm kiếm (SEARCH_RECORD).
• note: ghi chú mới (UPDATE_RECORD).

=== LƯU Ý PHÂN BIỆT (quan trọng) ===
• SET_ALERT (bật/tắt cảnh báo, nhắc nhở chi tiêu) ≠ SYSTEM_SETTING (giao diện sáng/tối, cài đặt app).
• SET_LIMIT (hạn mức/ngân sách) ≠ SET_GOAL/ADD_GOAL (mục tiêu tiết kiệm).
• SUGGEST_BUDGET (gợi ý ngân sách) ≠ REPORT_GENERAL (báo cáo/thống kê đã chi).
• Câu có số tiền nhưng chỉ MÔ TẢ đã tiêu ("hôm nay tiêu 2tr") → KHÔNG thuộc file này (thuộc intent_record).
• Không dùng nhãn Setting — dùng SYSTEM_SETTING.

=== PHÂN BỔ ~20.000 MẪU (gần đúng tỷ lệ) ===
REPORT_GENERAL ~2200, SET_LIMIT ~2200, SEARCH_RECORD ~1900,
SET_GOAL ~1450, SET_ALERT ~1400, SET_INCOME ~1400, SUGGEST_BUDGET ~1350,
SET_USERNAME ~1330, SET_TONE ~1330, ADD_GOAL ~1280, SYSTEM_SETTING ~1330

=== VÍ DỤ MẪU (format đúng — không copy nguyên văn) ===
text,intent,action_type,verb,category_code,value,goal_name,enabled,theme,verbal_style,time_range,query,note
Thêm 200k vào giới hạn đi lại,Action,SET_LIMIT,ADD,Transport,200000,,,,,,,
Đặt hạn mức ăn uống 2 triệu,Action,SET_LIMIT,SET,Food,2000000,,,,,,,
Bật cảnh báo vượt hạn mức cho giải trí,Action,SET_ALERT,,Entertainment,,,true,,,,,
Tắt thông báo chi tiêu,Action,SET_ALERT,,,,,false,,,,,
Tạo mục tiêu mua laptop mới 15 triệu,Action,SET_GOAL,,,15000000,mua laptop mới,,,,,,
Bù 2tr vào mục tiêu mua nhà,Action,ADD_GOAL,ADD,,2000000,mua nhà,,,,,,
Chuyển sang giao diện tối,Action,SYSTEM_SETTING,,,,,,dark,,,,
Đổi sang giọng nói châm chọc nhé,Action,SET_TONE,,,,,,,sarcastic,,,
Gọi mình là Khang nhé,Action,SET_USERNAME,,,Khang,,,,,,,
Lương tháng này 15 củ,Action,SET_INCOME,,,15000000,,,,,,,
Tìm giao dịch mua sắm trên 500k,Action,SEARCH_RECORD,,500000,,,,,,mua sắm,
Thống kê chi tiêu ăn uống tháng này,Action,REPORT_GENERAL,,,,,,,,tháng này,,
Gợi ý ngân sách cho tháng sau,Action,SUGGEST_BUDGET,,,,,,,,tháng sau,,

Sinh [N] dòng CSV (chỉ nội dung CSV, có header, không giải thích).
action_type cần sinh lần này: [GHI action_type HOẶC "mix theo tỷ lệ trên"].
```

**Script có sẵn:** `generate_action_dataset_api.py` (Gemini API, checkpoint, ~20k).

---

## 2. `intent_record.csv` — Ghi nhận chi tiêu / thu nhập (Record)

```
Bạn là chuyên gia gán nhãn NLU cho app quản lý chi tiêu Mimo (tiếng Việt, Gen Z, sinh viên, ngắn gọn).

Nhiệm vụ: xuất file 20k mẫu CSV intent_record.csv — mỗi dòng là câu GHI NHẬN một khoản chi hoặc thu (Record), KHÔNG phải lệnh hệ thống, KHÔNG phải chitchat.

=== CẤU TRÚC CSV ===
text,label,type,is_money

=== CỘT CHI TIẾT ===
• text (string): Câu mô tả giao dịch, thường có số tiền (19k, 2tr, 1.5 triệu…). Đa dạng: Grab, ShopeeFood, ký túc xá, học phí, lương, cafe, nhậu bạn bè…
• label (string): Mã danh mục — CHỈ một trong 18 giá trị sau:
  Food, Transport, Housing, Essentials, Shopping, Beauty, Health, Education,
  Entertainment, Social, Investment, Others,
  Salary, Bonus, Business, Debt, Charity, Savings
  (4 nhãn cuối thường đi với thu nhập: Salary, Bonus, Business, Investment…)
• type (string): expense | income (lowercase). Phần lớn là expense.
• is_money (int): 1 nếu câu có số tiền rõ ràng; 0 nếu không có số (hiếm).

=== QUY TẮC GÁN label ===
• Food: mua/ăn/uống mang về, đồ ăn, trà sữa MUA VỀ, cơm, phở (không đi chơi).
• Entertainment: đi chơi, nhậu, Netflix, game, cafe/hẹn hò/xã giao ("đi cf với bạn", "hẹn quán cafe với bồ").
• Transport: Grab, xăng, vé xe, gửi xe.
• Housing: tiền nhà, ký túc, điện nước, wifi.
• Essentials: đồ dùng hàng ngày, vệ sinh, thuốc OTC nhẹ.
• Shopping: quần áo, điện thoại, đồ online.
• Health: khám, thuốc bệnh, bảo hiểm y tế.
• Education: học phí, sách, khóa học.
• Social: quà tặng, đám cưới, liên hoan (không nhầm Entertainment).
• Investment: vàng, cổ phiếu, crypto (chi ra để đầu tư).
• Others: không khớp danh mục trên.
• Salary / Bonus / Business / Debt / Charity / Savings: thu nhập hoặc các loại đặc biệt — set type=income khi là tiền vào.

=== LƯU Ý PHÂN BIỆT (quan trọng) ===
• "Mua cafe 25k" → Food (mua đồ uống).
• "Đi cafe với bạn 50k" → Entertainment (đi chơi/xã giao).
• "Đặt hạn mức 2tr" / "thống kê tháng này" → KHÔNG thuộc file này.
• "Cảm ơn nha" / "Bạn khỏe không" → KHÔNG thuộc file này.
• Cùng một câu không xuất hiện 2 lần.

=== PHÂN BỐ GỢI Ý (dataset lớn) ===
Expense chiếm ~85–90%. Food, Transport, Entertainment, Shopping nhiều nhất.
Income (Salary, Bonus…) ~10–15%. is_money=1 cho >95% mẫu.

=== VÍ DỤ MẪU ===
text,label,type,is_money
Cà phê 28k,Food,expense,1
Trà sữa 19k,Food,expense,1
sữa tắm 12k,Essentials,expense,1
hớt tóc 100k,Beauty,expense,1
đi Grab 19k,Transport,expense,1
Order GrabFood cơm sườn 58k,Food,expense,1
Grab đi học 35k,Transport,expense,1
Tiền phòng ktx tháng 6 1tr5,Housing,expense,1
Mua cafe sữa đá 25k,Food,expense,1
Đi cà phê với bạn 19k,Entertainment,expense,1
hẹn đi cafe sữa đá với bồ 45k,Entertainment,expense,1
Tối đi nhậu bạn bè 230k,Entertainment,expense,1
Netflix tháng 109k,Entertainment,expense,1
Lương tháng 6 về 12tr,Salary,income,1
Thưởng dự án 2tr,Bonus,income,1
Freelance design 3tr5,Business,income,1
Chi tiêu linh tinh không nhớ,Others,expense,0

Sinh [N] dòng CSV (chỉ CSV + header). Ưu tiên label: [GHI label HOẶC "mix expense/income"].
```

**Script tham khảo:** `gemini_augment_record.py`, `improve_datasets.py`.

---

## 3. `intent_chitchat.csv` — Trò chuyện (Chitchat)

```
Bạn là chuyên gia gán nhãn NLU cho app quản lý chi tiêu Mimo (tiếng Việt, Gen Z).

Nhiệm vụ: sinh file CSV intent_chitchat.csv — câu TRÒ CHUYỆN với bot, KHÔNG ghi chi tiêu, KHÔNG ra lệnh hệ thống.

=== CẤU TRÚC CSV ===
text,intent,sentiment

=== CỘT CHI TIẾT ===
• text (string): Hỏi thăm, cảm ơn, chửi bới nhẹ, khen app, hỏi bot là ai, chào tạm biệt, tâm sự… Không có lệnh "đặt hạn mức", "xóa giao dịch", "thống kê".
• intent (string): LUÔN Chitchat (chữ C hoa, còn lại thường).
• sentiment (string): CHỈ một trong Positive | Neutral | Negative
  - Positive: cảm ơn, khen, vui, hài, yêu thích app.
  - Negative: buồn, mệt, chán, stress, bực (không toxic quá mức).
  - Neutral: hỏi thông tin, chào hỏi, hỏi bot làm gì, không rõ cảm xúc.

=== LƯU Ý ===
• Câu có số tiền + hành vi chi tiêu ("ăn trưa 50k") → intent_record, KHÔNG đưa vào file này.
• Câu lệnh app ("bật dark mode", "gợi ý ngân sách") → intent_action, KHÔNG đưa vào file này.
• Đa dạng: có/không dấu, teencode, "tui/mình/bro", emoji text.
• Không trùng câu.

=== PHÂN BỐ GỢI Ý ===
Neutral ~50%, Positive ~30%, Negative ~20%.

=== VÍ DỤ MẪU ===
text,intent,sentiment
Chào Mimo nha,Chitchat,Neutral
Cảm ơn bot nhiều lắm,Chitchat,Positive
Buồn quá không ai hiểu,Chitchat,Negative
Bạn là ai vậy,Chitchat,Neutral
App xịn quá trời,Chitchat,Positive
Mệt deadline dí cổ quá,Chitchat,Negative
Bye nha mai gặp lại,Chitchat,Positive
Bot có biết kể chuyện cười không,Chitchat,Neutral

Sinh [N] dòng CSV (chỉ CSV + header). sentiment cần: [Positive/Neutral/Negative hoặc "mix theo tỷ lệ"].
```

---

## Sau khi sinh xong

| Bước | Lệnh / việc cần làm |
|------|---------------------|
| Action slots | `python text_nlu/datasets/label_action_slots.py` |
| Train | `python text_nlu/train/retrain_all.py` (hoặc Kaggle) |
| Action API | `python text_nlu/datasets/generate_action_dataset_api.py --total 20000 --resume` |
