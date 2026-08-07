import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'api')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'nlu')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.nlu_service import get_nlu_service
from app.schemas.nlu import NLURequest

def main():
    print("="*60)
    print(" KIỂM TRA 10 CÂU CÓ DẤU - MÔ HÌNH PHOBERT MỚI (L4 GPU)")
    print("="*60)
    
    # Ép sử dụng PhoBERT
    os.environ["NLU_BACKEND"] = "phobert"
    
    # Load model
    service = get_nlu_service()
    service.reload()
    
    test_sentences = [
        "Hôm qua tui đi ăn lẩu thái Tomyum hết 500k",
        "Tháng này phải trả tiền điện nước 2 triệu nha",
        "Đóng học phí cho con ở trường quốc tế 15tr",
        "Vay mẹ 50 triệu mua xe SH",
        "Nhắc nợ thằng Tèo trả 500k tiền chầu nhậu tuần trước",
        "Tiết kiệm 5 triệu bỏ heo cuối năm đi du lịch",
        "Trúng số vietlott được 50 triệu",
        "Được sếp thưởng nóng 2 củ vì làm tốt",
        "Hello app, hôm nay trời đẹp quá",
        "Cậu có thể giúp tôi quản lý chi tiêu được không"
    ]
    
    for text in test_sentences:
        req = NLURequest(text=text)
        res = service.infer(req)
        print(f"Câu: '{text}'")
        print(f"  - Intent:   {res.intent}")
        print(f"  - Category: {res.category}")
        print(f"  - Action:   {res.action_type}")
        print(f"  - Record:   {res.record_type}")
        print("-" * 60)

if __name__ == "__main__":
    main()
