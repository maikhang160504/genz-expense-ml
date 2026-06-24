import pandas as pd
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION_CSV = ROOT / "datasets" / "intent_action.csv"

def generate_set_tone():
    prefixes = ["bật chế độ", "đổi sang giọng", "nói chuyện kiểu", "chuyển style sang", "cho mimo nói chuyện kiểu", "đổi phong cách nói chuyện thành", "đổi giọng nói thành", "nói giọng", "đổi giọng", "cài giọng nói", "doi giong", "noi chuyen kieu", "chuyen sang giong"]
    tones = ["dễ thương", "hài hước", "nghiêm túc", "châm chọc", "dạn dĩ", "vui vẻ", "đồng cảm", "đanh đá", "dịu dàng", "nghiêm khắc", "mẹ chồng", "người yêu", "de thuong", "hai huoc", "nghiem tuc", "cham choc", "dan di", "vui ve", "dong cam"]
    suffixes = ["nha Mimo", "giùm mình", "đi", "nhé", "nha", "coi", "thử xem", "nhe", "nha bạn", ""]
    
    rows = []
    for _ in range(230):
        t = f"{random.choice(prefixes)} {random.choice(tones)} {random.choice(suffixes)}".strip()
        rows.append({"text": t, "intent": "Action", "action_type": "SET_TONE"})
    return rows

def generate_export_data():
    prefixes = ["xuất báo cáo", "export dữ liệu", "gửi báo cáo", "gửi file excel", "tải sao kê", "xuất file csv", "gửi sao kê chi tiêu", "xuất excel", "xuất dữ liệu", "xuat bao cao", "gui file excel", "tai sao ke", "xuat file csv"]
    periods = ["tháng này", "tháng trước", "tuần này", "tuần trước", "năm nay", "hôm nay", "hôm qua", "quý này", "3 tháng qua", "thang nay", "thang truoc", "tuan nay", "tuan truoc", "nam nay", "hom nay", "hom qua"]
    suffixes = ["qua email cho tôi", "vào hòm thư", "qua gmail nhé Mimo", "về email", "để mình xem", "nha", "qua mail nhe", "di", "nhe", ""]
    
    rows = []
    for _ in range(230):
        t = f"{random.choice(prefixes)} {random.choice(periods)} {random.choice(suffixes)}".strip()
        rows.append({"text": t, "intent": "Action", "action_type": "EXPORT_DATA"})
    return rows

def generate_set_username():
    prefixes = ["gọi mình là", "gọi tôi là", "tên mình là", "đổi tên mình thành", "hãy gọi tớ là", "đổi tên hiển thị thành", "tên tôi là", "gọi tớ là", "goi minh la", "goi toi la", "ten minh la", "doi ten thanh"]
    names = ["An", "Khang", "Vy", "Linh", "Nam", "Dũng", "Vy", "Trang", "Minh", "Huy", "Hoàng", "Tùng", "Thảo", "Hà", "Phương", "Phúc", "Kiệt", "Dương", "Ngọc", "Sơn", "Lâm", "Hải", "Khánh"]
    suffixes = ["nhé Mimo", "nha Mimo", "nhe", "nha", "đi", "nhé", "nha bạn", ""]
    
    rows = []
    for _ in range(220):
        t = f"{random.choice(prefixes)} {random.choice(names)} {random.choice(suffixes)}".strip()
        rows.append({"text": t, "intent": "Action", "action_type": "SET_USERNAME"})
    return rows

def generate_set_income():
    prefixes = ["cài đặt thu nhập hàng tháng là", "thu nhập của mình là", "lương cứng mỗi tháng của tôi là", "đổi thu nhập tháng thành", "cập nhật thu nhập là", "thu nhập mỗi tháng", "lương tháng của mình là", "cai thu nhap", "thu nhap cua minh la", "luong thang la"]
    amounts = ["10 triệu", "15tr", "20 triệu", "8.5tr", "5000k", "12 triệu", "30tr", "7 củ", "10 củ", "15 củ", "25 củ", "5tr", "8tr", "10 triệu đồng", "20tr"]
    suffixes = ["nhé", "nha Mimo", "nha", "đi", "nhé", ""]
    
    rows = []
    for _ in range(190):
        t = f"{random.choice(prefixes)} {random.choice(amounts)} {random.choice(suffixes)}".strip()
        rows.append({"text": t, "intent": "Action", "action_type": "SET_INCOME"})
    return rows

def generate_delete_record():
    phrases = [
        "xóa giao dịch vừa rồi", "hủy khoản chi vừa ghi", "xóa hộ mình giao dịch gần nhất", "xóa giao dịch lúc nãy", "xóa lịch sử vừa rồi đi",
        "bỏ giao dịch vừa nhập", "hủy giao dịch gần nhất", "xóa cái vừa nhập", "xóa khoản chi vừa rồi", "hủy cái vừa ghi", "xóa bill vừa rồi",
        "xoa giao dich vua roi", "huy khoan chi vua ghi", "xoa ho minh giao dich gan nhat", "xoa cai vua nhap", "huy cai vua ghi",
        "bỏ khoản chi tiêu vừa ghi", "hủy hóa đơn vừa rồi", "xóa hóa đơn gần nhất", "hủy giao dịch vừa lưu", "xóa giao dịch vừa lưu",
        "xóa lịch sử chi tiêu vừa rồi", "hủy ghi chép vừa rồi", "bỏ giao dịch lúc nãy đi", "xóa cái bill vừa nhập", "hủy hóa đơn vừa ghi"
    ]
    suffixes = ["đi", "nha Mimo", "nhé Mimo", "nhé", "nha", "giùm cái", "nhe Mimo", ""]
    
    rows = []
    for _ in range(170):
        t = f"{random.choice(phrases)} {random.choice(suffixes)}".strip()
        rows.append({"text": t, "intent": "Action", "action_type": "DELETE_RECORD"})
    return rows

def main():
    print("Reading intent_action.csv...")
    df = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    
    initial_counts = df["action_type"].value_counts()
    print("Initial action_type distribution:")
    print(initial_counts)
    
    # Generate new rows
    new_rows = []
    new_rows.extend(generate_set_tone())
    new_rows.extend(generate_set_username())
    new_rows.extend(generate_set_income())
    new_rows.extend(generate_delete_record())
    
    new_df = pd.DataFrame(new_rows)
    
    # Remove duplicates from new generated rows that might already exist
    existing_texts = set(df["text"].astype(str).str.strip().str.lower())
    filtered_new_rows = []
    for _, row in new_df.iterrows():
        txt = str(row["text"]).strip().lower()
        if txt not in existing_texts:
            existing_texts.add(txt)
            filtered_new_rows.append(row)
            
    filtered_new_df = pd.DataFrame(filtered_new_rows)
    print(f"Generated {len(new_df)} rows. After filtering duplicates, adding {len(filtered_new_df)} rows.")
    
    balanced_df = pd.concat([df, filtered_new_df], ignore_index=True)
    balanced_df.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")
    
    final_counts = balanced_df["action_type"].value_counts()
    print("\nBalanced action_type distribution:")
    print(final_counts)
    print(f"Total lines in intent_action.csv: {len(balanced_df)}")

if __name__ == "__main__":
    main()
