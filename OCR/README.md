# Module OCR & Key Information Extraction (KIE)

Thư mục này chứa toàn bộ mã nguồn, mô hình huấn luyện, dữ liệu cấu hình và pipeline nhận dạng hóa đơn (OCR) tích hợp trích xuất thông tin KIE (Key Information Extraction) phục vụ cho dự án MoneyStory.

---

## 🏛️ Cấu Trúc Thư Mục Phân Loại

Các thành phần được phân chia rõ ràng theo chức năng cốt lõi:

```text
OCR/
├── src/                     # Mã nguồn xử lý chính của pipeline
│   └── receipt_ocr/         # Package python thực thi logic OCR + KIE
│       ├── mcocr_rotation/  # Thuật toán xử lý nghiêng (MobileNetV3)
│       ├── hybrid_pipeline.py # Định nghĩa pipeline chính kết hợp các bước
│       ├── pick_kie.py      # Logic gán nhãn thực thể PICK
│       ├── pick_kie_inference.py # Thực thi dự đoán PICK KIE
│       └── receipt_nlu.py   # Phân loại danh mục bill & kiểm chéo số tiền
│
├── models/                  # Nơi lưu trữ weights của các mô hình
│   ├── paddleocr/           # Weights phát hiện chữ (Text Detection)
│   ├── vietocr/             # Weights nhận diện chữ (vgg_transformer / vietocr_receipt)
│   ├── pick_kie/            # Weights trích xuất thực thể (model_best.pth)
│   └── rotation_corrector/  # Weights MobileNetV3 chỉnh góc xoay 0-180 độ
│
├── vendor/                  # Mã nguồn bên thứ ba được đóng gói cục bộ (Vendored)
│   └── pick/                # Toàn bộ source code train/infer của model PICK KIE
│
├── kaggle/                  # Các kernel phục vụ huấn luyện trên Kaggle GPU
│   ├── kernels/             # Các tệp IPYNB notebook và metadata để push lên Kaggle
│   │   ├── train-pick-kie/    # Kernel huấn luyện mới model PICK
│   │   └── retrain-pick-kie/  # Kernel huấn luyện lại model PICK bằng data WebAdmin
│   ├── pick_kaggle_common.py  # Hàm helper dùng chung trên môi trường Kaggle
│   └── build_pick_kaggle_notebooks.py # Script sinh tự động các notebook
│
├── verified_ocr_labels/     # Dữ liệu xuất bản nhãn đã duyệt từ WebAdmin
│   ├── incremental/         # Data nhãn (.tsv) và ảnh (.jpg) được gom mới
│   └── kaggle_upload/       # Thư mục tạm nén zip để đồng bộ với Kaggle Dataset
│
├── manifests/               # Quản lý phiên bản mô hình đang hoạt động
│   └── ocr_models.json      # File manifest định nghĩa đường dẫn weight active
│
├── tests/                   # Bộ kiểm thử tự động (Unit Tests)
│   └── test_hybrid_ocr.py   # Kiểm tra tính toàn vẹn và độ chính xác của pipeline
│
├── tools/                   # Các script chạy thử nhanh (Smoke Tests)
│   └── _smoke_pick_kie.py   # Chạy thử nhanh mô hình PICK KIE
│
└── requirements.txt         # Các dependencies cần thiết cho phân hệ OCR
```

---

## 🔄 Luồng Hoạt Động Của Pipeline OCR

Dữ liệu hóa đơn đi qua 5 giai đoạn xử lý khép kín:

```
[ Ảnh hóa đơn thô ]
        │
        ▼
[ Giai đoạn 1: Page Rotation ] ─────► Sử dụng MobileNetV3 xoay ảnh thẳng (0 hoặc 180 độ)
        │
        ▼
[ Giai đoạn 2: Text Detection ] ────► PaddleOCR định vị tọa độ các hộp chữ (Bbox)
        │
        ▼
[ Giai đoạn 3: Text Recognition ] ──► VietOCR đọc nội dung chữ tiếng Việt bên trong Bbox
        │
        ▼
[ Giai đoạn 4: PICK KIE ] ──────────► Mô hình đồ thị (Graph) gán nhãn thực thể cho từng Bbox
        │                             (SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, OTHER)
        ▼
[ Giai đoạn 5: NLU Fusion ] ────────► Kiểm chéo tổng tiền với tổng các món hàng (chặn lỗi Digit Drop)
                                      và chuyển text hóa đơn vào NLU Category Model để đoán danh mục.
```

---

## 🛠️ Hướng Dẫn Vận Hành

### 1. Chạy Kiểm Thử Tự Động (Unit Tests)
Để đảm bảo code chỉnh sửa không làm vỡ pipeline, kích hoạt virtual environment và chạy:
```bash
# Kích hoạt venv
.venv\Scripts\activate

# Chạy test suite
pytest OCR/tests/test_hybrid_ocr.py -v
```

### 2. Tái Tạo Notebook Để Đẩy Lên Kaggle
Mỗi khi thay đổi helper dùng chung `pick_kaggle_common.py`, cần build lại các notebook:
```bash
python OCR/kaggle/kernels/build_pick_kaggle_notebooks.py
```
Sau đó, các file `.ipynb` và `kernel-metadata.json` sẽ tự động cập nhật trong thư mục `OCR/kaggle/kernels/`, sẵn sàng để push lên Kaggle qua WebAdmin hoặc Kaggle CLI.

### 3. Nguyên Tắc Độc Lập
* Toàn bộ mã nguồn PICK được vendored cục bộ tại `OCR/vendor/pick`.
* Hệ thống **không phụ thuộc** vào bất kỳ repo ngoài nào khác tại runtime (bao gồm cả repo gốc `MC_OCR` cũ).
* Trong môi trường Cloud, hệ thống tự động tải ảnh hóa đơn từ Cloudflare R2 về thư mục `verified_ocr_labels/incremental/images` trước khi nén zip gửi lên Kaggle.
