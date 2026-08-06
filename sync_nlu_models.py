import os
import subprocess
import sys

def sync_nlu_models():
    print("🚀 Bắt đầu đồng bộ thư mục nlu_models từ Modal về máy local...")
    
    local_dir = "models"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"📂 Đã tạo thư mục '{local_dir}'")
        
    # Sử dụng command dạng string với shell=True trên Windows để nhận diện đúng biến môi trường và file .exe/.cmd
    cmd = "modal volume get expense-ocr-nlu-storage nlu_models/* models/"
    
    try:
        print(f"🔄 Đang thực thi: {cmd}")
        print("⏳ Quá trình này có thể mất vài phút tùy vào tốc độ mạng...")
        
        # Chạy subprocess và in trực tiếp output ra màn hình
        process = subprocess.run(cmd, shell=True, check=True)
        
        print("\n✅ ĐỒNG BỘ THÀNH CÔNG!")
        print(f"Tất cả các tệp (model, json) đã được lưu an toàn vào: {os.path.abspath(local_dir)}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Đã xảy ra lỗi khi đồng bộ! Mã lỗi: {e.returncode}")
        sys.exit(1)
        
if __name__ == "__main__":
    sync_nlu_models()
