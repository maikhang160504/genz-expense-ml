"""Generate rich dataset for saving, challenge, and loan tools and merge into intent_action.csv."""
from __future__ import annotations

import csv
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "intent_action.csv"

# Columns defined in action_slot_columns.ALL_COLUMNS
COLUMNS = [
    "text", "intent", "action_type", "verb", "category_code",
    "value", "goal_name", "enabled", "theme", "verbal_style",
    "time_range", "query", "note"
]

SAMPLES = [
    # Tiết kiệm cá nhân - SET_GOAL
    ("Tạo mục tiêu tiết kiệm mua xe máy 45 triệu", "Action", "SET_GOAL", "SET", "", "45000000", "Tiết kiệm mua xe máy", "", "", "", "", "", "saving"),
    ("Mở tiết kiệm mua laptop mới 25 triệu đồng", "Action", "SET_GOAL", "SET", "", "25000000", "Tiết kiệm mua laptop", "", "", "", "", "", "saving"),
    ("Tạo quỹ tiết kiệm du lịch Đà Nẵng 10 triệu", "Action", "SET_GOAL", "SET", "", "10000000", "Tiết kiệm du lịch Đà Nẵng", "", "", "", "", "", "saving"),
    ("Đặt mục tiêu tiết kiệm mua điện thoại 15 triệu", "Action", "SET_GOAL", "SET", "", "15000000", "Tiết kiệm mua điện thoại", "", "", "", "", "", "saving"),
    ("Tạo mục tiêu tiết kiệm quỹ dự phòng 50 triệu", "Action", "SET_GOAL", "SET", "", "50000000", "Quỹ dự phòng", "", "", "", "", "", "saving"),
    ("Tạo mục tiêu tiết kiệm mua quà sinh nhật mẹ 3 triệu", "Action", "SET_GOAL", "SET", "", "3000000", "Quà sinh nhật mẹ", "", "", "", "", "", "saving"),
    ("Mở mục tiêu tiết kiệm mua xe ô tô 500 triệu", "Action", "SET_GOAL", "SET", "", "500000000", "Tiết kiệm mua ô tô", "", "", "", "", "", "saving"),
    ("Thiết lập mục tiêu tiết kiệm cưới vợ 100 triệu", "Action", "SET_GOAL", "SET", "", "100000000", "Tiết kiệm cưới vợ", "", "", "", "", "", "saving"),
    ("Tạo tiết kiệm mua máy ảnh 18 triệu", "Action", "SET_GOAL", "SET", "", "18000000", "Mua máy ảnh", "", "", "", "", "", "saving"),
    ("Tạo mục tiêu tiết kiệm sửa nhà 80 triệu", "Action", "SET_GOAL", "SET", "", "80000000", "Tiết kiệm sửa nhà", "", "", "", "", "", "saving"),

    # Tiết kiệm tập thể (rủ thêm người tham gia) - SET_GOAL
    ("Tạo mục tiêu tiết kiệm tập thể du lịch Thái Lan cùng nhóm bạn 30 triệu", "Action", "SET_GOAL", "SET", "", "30000000", "Tiết kiệm tập thể du lịch Thái Lan", "", "", "", "", "", "saving"),
    ("Mở quỹ tiết kiệm chung nhóm 4 người đi Phú Quốc 20 triệu", "Action", "SET_GOAL", "SET", "", "20000000", "Tiết kiệm chung Phú Quốc", "", "", "", "", "", "saving"),
    ("Tạo tiết kiệm tập thể cùng bạn bè mua quà cưới 5 triệu", "Action", "SET_GOAL", "SET", "", "5000000", "Tiết kiệm tập thể quà cưới", "", "", "", "", "", "saving"),
    ("Tạo mục tiêu tiết kiệm nhóm đi cắm trại 6 triệu rủ thêm người tham gia", "Action", "SET_GOAL", "SET", "", "6000000", "Tiết kiệm nhóm cắm trại", "", "", "", "", "", "saving"),
    ("Tạo tiết kiệm tập thể quỹ lớp 15 triệu", "Action", "SET_GOAL", "SET", "", "15000000", "Tiết kiệm quỹ lớp", "", "", "", "", "", "saving"),
    ("Mở quỹ tiết kiệm chung gia đình 50 triệu đi du lịch hè", "Action", "SET_GOAL", "SET", "", "50000000", "Quỹ du lịch gia đình", "", "", "", "", "", "saving"),

    # Tiết kiệm - ADD_GOAL (nạp tiền / đóng góp)
    ("Đóng góp 2 triệu vào tiết kiệm mua xe máy", "Action", "ADD_GOAL", "ADD", "", "2000000", "Tiết kiệm mua xe máy", "", "", "", "", "", "saving"),
    ("Nạp thêm 1 triệu vào quỹ tiết kiệm du lịch", "Action", "ADD_GOAL", "ADD", "", "1000000", "Tiết kiệm du lịch", "", "", "", "", "", "saving"),
    ("Thêm 500k vào mục tiêu tiết kiệm mua laptop", "Action", "ADD_GOAL", "ADD", "", "500000", "Tiết kiệm mua laptop", "", "", "", "", "", "saving"),
    ("Bỏ 3 triệu vào quỹ tiết kiệm mua nhà", "Action", "ADD_GOAL", "ADD", "", "3000000", "Tiết kiệm mua nhà", "", "", "", "", "", "saving"),
    ("Đóng 500 ngàn vào tiết kiệm tập thể nhóm đi Đà Lạt", "Action", "ADD_GOAL", "ADD", "", "500000", "Tiết kiệm nhóm đi Đà Lạt", "", "", "", "", "", "saving"),
    ("Góp thêm 1 triệu vào tiết kiệm chung gia đình", "Action", "ADD_GOAL", "ADD", "", "1000000", "Tiết kiệm chung gia đình", "", "", "", "", "", "saving"),
    ("Thêm 2 triệu vào mục tiêu tiết kiệm sửa nhà", "Action", "ADD_GOAL", "ADD", "", "2000000", "Tiết kiệm sửa nhà", "", "", "", "", "", "saving"),
    ("Nạp 300k vào quỹ tiết kiệm du lịch hè", "Action", "ADD_GOAL", "ADD", "", "300000", "Tiết kiệm du lịch hè", "", "", "", "", "", "saving"),

    # Thử thách cá nhân - SET_GOAL
    ("Tạo thử thách không uống trà sữa 30 ngày tiết kiệm 600k", "Action", "SET_GOAL", "SET", "", "600000", "Thử thách không uống trà sữa", "", "", "", "", "", "challenge"),
    ("Mở thử thách tự nấu ăn buổi trưa 14 ngày tiết kiệm 700k", "Action", "SET_GOAL", "SET", "", "700000", "Thử thách tự nấu ăn", "", "", "", "", "", "challenge"),
    ("Tạo thử thách không ăn vặt 1 tháng tiết kiệm 500k", "Action", "SET_GOAL", "SET", "", "500000", "Thử thách không ăn vặt", "", "", "", "", "", "challenge"),
    ("Mở thử thách tiết kiệm mỗi ngày 20k trong 30 ngày tổng 600k", "Action", "SET_GOAL", "SET", "", "600000", "Thử thách tiết kiệm mỗi ngày", "", "", "", "", "", "challenge"),
    ("Tạo thử thách cai mua sắm quần áo 2 tháng tiết kiệm 2 triệu", "Action", "SET_GOAL", "SET", "", "2000000", "Thử thách cai mua sắm", "", "", "", "", "", "challenge"),
    ("Tạo thử thách không đi cà phê sang chảnh tiết kiệm 1 triệu", "Action", "SET_GOAL", "SET", "", "1000000", "Thử thách không cà phê sang chảnh", "", "", "", "", "", "challenge"),

    # Thử thách tập thể (cùng tham gia, tiến độ riêng) - SET_GOAL
    ("Tạo thử thách tập thể không uống trà sữa cùng nhóm bạn tiết kiệm 1 triệu", "Action", "SET_GOAL", "SET", "", "1000000", "Thử thách tập thể không uống trà sữa", "", "", "", "", "", "challenge"),
    ("Mở thử thách chạy bộ mỗi ngày cùng nhóm bạn tiết kiệm 500k", "Action", "SET_GOAL", "SET", "", "500000", "Thử thách chạy bộ", "", "", "", "", "", "challenge"),
    ("Tạo thử thách nhóm mang cơm trưa văn phòng tiết kiệm 1 triệu rưỡi", "Action", "SET_GOAL", "SET", "", "1500000", "Thử thách mang cơm trưa", "", "", "", "", "", "challenge"),
    ("Rủ nhóm bạn tham gia thử thách tiết kiệm 30 ngày 900k", "Action", "SET_GOAL", "SET", "", "900000", "Thử thách tiết kiệm 30 ngày", "", "", "", "", "", "challenge"),

    # Thử thách - ADD_GOAL (cập nhật tiến độ / đóng góp)
    ("Cập nhật tiến độ thử thách không uống trà sữa thêm 50k", "Action", "ADD_GOAL", "ADD", "", "50000", "Thử thách không uống trà sữa", "", "", "", "", "", "challenge"),
    ("Đóng góp 100k vào tiến độ thử thách tự nấu ăn", "Action", "ADD_GOAL", "ADD", "", "100000", "Thử thách tự nấu ăn", "", "", "", "", "", "challenge"),
    ("Nạp 20k hoàn thành thử thách tiết kiệm hôm nay", "Action", "ADD_GOAL", "ADD", "", "20000", "Thử thách tiết kiệm hôm nay", "", "", "", "", "", "challenge"),
    ("Cập nhật tiến độ thử thách mang cơm trưa 40k", "Action", "ADD_GOAL", "ADD", "", "40000", "Thử thách mang cơm trưa", "", "", "", "", "", "challenge"),
    ("Thêm 50k vào thử thách không ăn vặt", "Action", "ADD_GOAL", "ADD", "", "50000", "Thử thách không ăn vặt", "", "", "", "", "", "challenge"),

    # Vay mượn (nhắc hẹn cho vay / đi vay) - SET_GOAL
    ("Tạo nhắc hẹn cho Nam vay 5 triệu hạn chót 30/8", "Action", "SET_GOAL", "SET", "", "5000000", "Cho Nam vay", "", "", "", "", "", "loan"),
    ("Nhắc hẹn mượn Hùng 2 triệu trả vào ngày 15 tháng 9", "Action", "SET_GOAL", "SET", "", "2000000", "Mượn Hùng", "", "", "", "", "", "loan"),
    ("Tạo khoản vay 10 triệu cho Linh hạn 20/10", "Action", "SET_GOAL", "SET", "", "10000000", "Cho Linh vay", "", "", "", "", "", "loan"),
    ("Ghi nhớ cho Tuấn vay 3 triệu", "Action", "SET_GOAL", "SET", "", "3000000", "Cho Tuấn vay", "", "", "", "", "", "loan"),
    ("Tạo nhắc hẹn mượn Mai 1 triệu rưỡi tuần sau trả", "Action", "SET_GOAL", "SET", "", "1500000", "Mượn Mai", "", "", "", "", "", "loan"),
    ("Tạo nhắc hẹn cho Minh vay 7 triệu hạn cuối tháng này", "Action", "SET_GOAL", "SET", "", "7000000", "Cho Minh vay", "", "", "", "", "", "loan"),
    ("Nhắc tôi trả khoản mượn Thành 4 triệu vào 10/9", "Action", "SET_GOAL", "SET", "", "4000000", "Mượn Thành", "", "", "", "", "", "loan"),
    ("Tạo lời nhắc cho vay 15 triệu với anh Hoàng", "Action", "SET_GOAL", "SET", "", "15000000", "Cho anh Hoàng vay", "", "", "", "", "", "loan"),
    ("Tạo nhắc hẹn đi vay Ngân hàng 20 triệu ngày trả 15 hàng tháng", "Action", "SET_GOAL", "SET", "", "20000000", "Vay Ngân hàng", "", "", "", "", "", "loan"),
    ("Ghi chú cho vay 500 ngàn bạn Khang", "Action", "SET_GOAL", "SET", "", "500000", "Cho Khang vay", "", "", "", "", "", "loan"),
]


def main():
    existing_texts = set()
    rows = []
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_texts.add(r.get("text", "").strip())
                rows.append(r)

    added = 0
    for sample in SAMPLES:
        text = sample[0].strip()
        if text in existing_texts:
            continue
        row_dict = {col: "" for col in COLUMNS}
        for idx, col in enumerate(COLUMNS):
            row_dict[col] = sample[idx]
        rows.append(row_dict)
        existing_texts.add(text)
        added += 1

    with open(DATASET_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"SUCCESS: Appended {added} financial tool samples to {DATASET_PATH.name} (Total rows: {len(rows)})")


if __name__ == "__main__":
    main()
