import os
import subprocess
import sys

def sync_nlu_models():
    print("🚀 Bắt đầu đồng bộ thư mục nlu_models từ Modal về máy local...")
    
    local_dir = "models"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"📂 Đã tạo thư mục '{local_dir}'")
        
    # Tải thư mục nlu_models từ Modal về thư mục hiện tại (sẽ sinh ra thư mục nlu_models/)
    cmd = "modal volume get expense-ocr-nlu-storage nlu_models/ . --force"
    
    try:
        print(f"🔄 Đang thực thi: {cmd}")
        print("⏳ Quá trình này có thể mất vài phút tùy vào tốc độ mạng...")
        
        process = subprocess.run(cmd, shell=True, check=True)
        
        # Di chuyển toàn bộ file từ nlu_models/ sang models/
        import shutil
        if os.path.exists("nlu_models"):
            for file_name in os.listdir("nlu_models"):
                shutil.move(os.path.join("nlu_models", file_name), os.path.join(local_dir, file_name))
            os.rmdir("nlu_models") # Xóa thư mục rỗng
            
        print("\n✅ ĐỒNG BỘ THÀNH CÔNG!")
        print(f"Tất cả các tệp (model, json) đã được lưu an toàn vào: {os.path.abspath(local_dir)}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Đã xảy ra lỗi khi đồng bộ! Mã lỗi: {e.returncode}")
        sys.exit(1)
        
if __name__ == "__main__":
    sync_nlu_models()
