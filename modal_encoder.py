import modal
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

volume = modal.Volume.from_name("expense-ocr-nlu-storage", create_if_missing=True)
app = modal.App(name="expense-ocr-nlu-encoder")

# Định nghĩa môi trường Image (y hệt modal_app.py để tránh lỗi)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        "torch>=2.0.0", "torchvision", "transformers>=4.45.0"
    )
    .pip_install(
        "paddleocr==2.7.3", "paddlepaddle-gpu==2.6.1", "vietocr==0.3.13", "opencv-python-headless",
        "accelerate>=0.30.0", "peft>=0.12.0", "bitsandbytes", "einops", "sentencepiece", "triton"
    )
    .pip_install("setuptools_scm<8.0.0") # FIX: Bắt buộc hạ cấp setuptools_scm để seqeval không bị lỗi build
    .pip_install(
        "fastapi==0.115.0", "google-genai", "uvicorn[standard]==0.30.6", "pydantic==2.9.2",
        "pydantic-settings==2.5.2", "python-multipart==0.0.9", "python-dotenv==1.0.1", "httpx==0.27.2",
        "pandas==2.2.2", "numpy==1.26.4", "joblib==1.4.2", "scikit-learn==1.5.2", "pyvi==0.1.1", 
        "psycopg2-binary", "spacy==3.8.10", "overrides", "prefetch-generator", "torchtext==0.6.0", 
        "Levenshtein", "yacs", "torchsummary", "datasets", "seqeval"
    )
    .run_commands(
        "python -c \""
        "from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForTokenClassification; "
        "AutoTokenizer.from_pretrained('vinai/phobert-base'); "
        "AutoModel.from_pretrained('vinai/phobert-base'); "
        "AutoProcessor.from_pretrained('microsoft/layoutlmv3-base'); "
        "AutoModelForTokenClassification.from_pretrained('microsoft/layoutlmv3-base'); "
        "\""
    )
    .add_local_dir(
        local_path=".", 
        remote_path="/workspace", 
        ignore=[".venv", "data", "exported", "__pycache__", ".git"], 
        copy=True
    )
)

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="l4", # Dùng GPU xịn L4 để train cực nhanh
    timeout=10800,
)
def train_encoder_modal():
    """Hàm huấn luyện độc lập chuyên dành cho Encoder trên Modal"""
    import sys
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/src/api")
    os.environ["EXPENSE_OCR_NLU_DIR"] = "/workspace"
    
    from app.routers.nlu import run_retraining
    
    print("Bắt đầu huấn luyện PhoBERT (Encoder)...")
    # Bước 1: Huấn luyện mô hình (nó sẽ ghi kết quả vào /workspace/text_nlu/models_new)
    # Vì add_local_dir đã copy sẵn intent_record.csv... nên nó sẽ tự động lấy dữ liệu mới nhất
    run_retraining(Path("/workspace"), target="encoder")
    
    # Bước 2: CHỮA LỖI LƯU TRỮ (Đưa từ /workspace tạm thời sang /storage vĩnh viễn)
    models_new = Path("/workspace/text_nlu/models_new")
    storage_models = Path("/storage/nlu_models")
    
    if models_new.exists():
        print(f"Đang đồng bộ mô hình mới vào thư mục vĩnh viễn {storage_models}...")
        
        # Xóa đúng các file/thư mục cũ tương ứng sẽ được cập nhật
        if storage_models.exists():
            for item in models_new.iterdir():
                target_path = storage_models / item.name
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path, ignore_errors=True)
                    else:
                        target_path.unlink(missing_ok=True)
                        
        # Đè trực tiếp các file mô hình mới lên bản hiện hành (giữ nguyên các file khác)
        shutil.copytree(models_new, storage_models, dirs_exist_ok=True)
        volume.commit()
        print("TẤT CẢ FILE ĐÃ ĐƯỢC LƯU THÀNH CÔNG VÀO MODAL VOLUME! 🎉")
    else:
        print("LỖI: Không tìm thấy thư mục models_new sau khi huấn luyện xong!")
