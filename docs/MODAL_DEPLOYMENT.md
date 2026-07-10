# Hướng dẫn triển khai và huấn luyện Serverless GPU trên Modal.com

Tài liệu này hướng dẫn chi tiết quy trình chuẩn bị dữ liệu, chạy thử nghiệm từng bước tiền xử lý, huấn luyện mô hình trích xuất thông tin khóa hóa đơn (LayoutLMv3 KIE) trên Cloud GPU, và triển khai FastAPI Web App phục vụ API chính thức thông qua nền tảng **Modal**.

---

## 1. Chuẩn bị môi trường cục bộ (Local)

1. Cài đặt thư viện Modal CLI và các thư viện hỗ trợ tại máy local:
   ```bash
   pip install modal python-Levenshtein
   ```
2. Thiết lập cấu hình và liên kết tài khoản Modal của bạn:
   ```bash
   modal setup
   ```
   *Lệnh này sẽ mở trình duyệt để bạn đăng nhập hoặc tạo tài khoản Modal (tài khoản mới được tặng $30 free credit mỗi tháng).*

---

## 2. Chuẩn bị dữ liệu

Trước khi tải dữ liệu lên Cloud để huấn luyện, chúng ta lưu trữ dữ liệu tại thư mục `data/` ở thư mục gốc của dự án.

```bash
modal volume put expense-ocr-nlu-storage d:\Luan-Van\Project\expense-ocr-nlu\data /data
```

1. Đặt thư mục dữ liệu thô tại: `data/mc_ocr_train/` và tệp nhãn tại `data/train_df.csv`.
2. Dữ liệu này sẽ tự động được sử dụng trong quá trình huấn luyện khi chạy lệnh trên Modal.

---

## 3. Chạy từng bước tiền xử lý trên Cloud (Debug độc lập)

Do quy trình xử lý ảnh hóa đơn bao gồm nhiều bước phức tạp (Detect chữ -> Xoay thẳng ảnh -> Nhận diện chữ -> Tạo đồ thị), hệ thống đã được chia nhỏ thành các hàm chạy độc lập trên Modal. Các kết quả trung gian sẽ tự động được lưu trữ và chia sẻ qua **Modal Volume (`/storage/output`)** để tránh việc phải chạy lại từ đầu nếu gặp lỗi.

Hãy chạy lần lượt các bước sau từ thư mục `expense-ocr-nlu`:

### Bước 1: Nhận diện vùng văn bản (Text Detector)
Tìm kiếm tọa độ của các hộp chữ trên ảnh sử dụng mô hình PaddleOCR (hệ thống sẽ tự động tải weights về container lúc chạy):
```bash
modal run modal_app.py::run_step_detector
```
*Đầu ra lưu tại: `/storage/output/text_detector/...`*

### Bước 2: Chỉnh góc xoay của hóa đơn (Rotation Corrector)
Xoay thẳng ảnh hóa đơn bị nghiêng hoặc ngược dựa trên mô hình MobileNetV3:
```bash
modal run modal_app.py::run_step_rotator
```
*Đầu ra lưu tại: `/storage/output/rotation_corrector/...`*

### Bước 3: Nhận dạng chữ viết (Text Classifier / OCR)
Sử dụng mô hình VietOCR (VGG19 Seq2Seq) để nhận dạng văn bản tiếng Việt bên trong các vùng chữ đã được cắt xoay:
```bash
modal run modal_app.py::run_step_classifier
```
*Đầu ra lưu tại: `/storage/output/text_classifier/...`*

### Bước 4: Tạo dữ liệu huấn luyện (LayoutLMv3 JSONL Dataset)
Khớp nhãn thực tế với các vùng chữ đã OCR, gán nhãn `"O"` cho các từ nền và tạo ra tệp huấn luyện dạng JSONL cho LayoutLMv3. 

*Đặc biệt: Bước này sẽ tự động phát hiện và trộn (merge) dữ liệu ảnh + nhãn thủ công đã xuất từ WebAdmin vào bộ dữ liệu huấn luyện.*
```bash
modal run modal_app.py::train_layoutlmv3_model
```

---

## 4. Huấn luyện mô hình LayoutLMv3 KIE trên GPU Cloud

### 4.1. Huấn luyện mô hình LayoutLMv3
Khi kích hoạt tiến trình huấn luyện, hệ thống sẽ tự động thực hiện các bước:
1. Chạy OCR trên toàn bộ tập train để ánh xạ nhãn thực tế và từ nền (PA1).
2. Lưu các tệp JSONL huấn luyện trực tiếp vào Modal Volume `/storage/layoutlmv3/jsonl`.
3. Bắt đầu quá trình huấn luyện bằng GPU Nvidia A10G/H100 với các tham số tối ưu (num_epochs=30, learning_rate=2e-5).

Chạy lệnh huấn luyện dạng tách biệt (detached) để tiến trình tự động chạy ngầm trên đám mây:
```bash
modal run --detach modal_app.py::train_layoutlmv3_model --num-epochs=30 --learning-rate=2e-5
```

### 4.2. Huấn luyện lại mô hình (Re-train) từ Web-Admin
Bạn cũng có thể kích hoạt trực tiếp từ giao diện quản trị Web-Admin thông qua nút **"Train LayoutLMv3"**, nút này sẽ gọi trực tiếp đến API của backend và kích hoạt tiến trình serverless huấn luyện trên Modal Cloud mà không cần dòng lệnh.

### 4.3. Tiến trình xử lý tự động của hệ thống
* Khởi chạy máy ảo GPU đám mây (Nvidia A10G/H100) trên Modal.
* Đọc dữ liệu huấn luyện và thực hiện fine-tune mô hình LayoutLMv3.
* Khi hoàn tất, tệp trọng số tốt nhất (`model_best.pth`) sẽ tự động được xuất ngược lại Volume lưu trữ đám mây của bạn tại `/storage/layoutlmv3/model_best.pth`.
* Tự động giải phóng máy ảo GPU để dừng tính phí.

---

## 5. Chạy thử nghiệm API phục vụ (Serve)

Để chạy thử nghiệm FastAPI server trên Cloud ở chế độ Dev (tự động cập nhật code mỗi khi bạn lưu file ở local):

```bash
modal serve modal_app.py
```
*Đầu ra sẽ hiển thị một URL HTTPS tạm thời dạng `https://<username>--expense-ocr-nlu-fastapi-app-dev.modal.run`.*

---

## 6. Triển khai API chính thức lên Cloud (Deploy)

Triển khai ứng dụng FastAPI hợp nhất (NLU + OCR + Retrain API) hoạt động vĩnh viễn dưới dạng Serverless (chỉ tốn chi phí khi có request gọi đến):

```bash
modal deploy modal_app.py
```

**Kết quả:**
* Modal sẽ trả về một URL HTTPS cố định dạng:
  `https://maikhang160504--expense-ocr-nlu-fastapi-app.modal.run`
* Mỗi khi WebApp gọi đến URL này, máy ảo sẽ tự động khởi động trong khoảng 1-2 giây, xử lý nhận dạng qua GPU T4 và trả kết quả.
* Bạn cần copy URL này cấu hình vào biến `AI_SERVICE_URL` ở tệp `.env` của backend Node.js.

---

## 7. Quy trình đồng bộ dữ liệu khi huấn luyện lại (Retrain) NLU & OCR

Để đảm bảo hiệu quả đồng bộ dữ liệu giữa máy phát triển cục bộ (Laptop) và đám mây (Modal Cloud Volume):

### 7.1. Đối với dữ liệu NLU (`intent_record.csv`)
Khi bạn thêm mẫu huấn luyện từ tính năng **Correction Clusters (Layer 2)** trên giao diện Admin:
1. **Lưu trữ cục bộ:** Backend Node.js (chạy local trên laptop) sẽ tự động ghi nhận và thêm các câu lệnh mới trực tiếp vào tệp dữ liệu cục bộ của bạn tại `expense-ocr-nlu/text_nlu/datasets/intent_record.csv`.
2. **Đồng bộ lên Cloud:** Để Cloud Container của Modal nhận diện được tập dữ liệu mới này trước khi bấm nút Train NLU trên giao diện, bạn chỉ cần thực thi lệnh sau từ terminal laptop:
   ```bash
   modal deploy modal_app.py
   ```
   *Nhờ cơ chế delta-upload thông minh của Modal, tiến trình này chỉ mất 3-5 giây để đẩy các thay đổi của file CSV lên Cloud và khởi tạo lại API.*
3. **Lưu trữ model sau train:** Sau khi huấn luyện NLU thành công, các mô hình phân lớp `.joblib` và mô hình NER spaCy sẽ được tự động đồng bộ vào Modal Volume `/storage/nlu_models/` để lưu trữ bền vững.

### 7.2. Đối với dữ liệu OCR (Ảnh và Nhãn hóa đơn)
Khi Admin phê duyệt và xuất dữ liệu sửa đổi từ trang duyệt hóa đơn (Approved -> Export):
1. **Lưu trữ Cloud-Native:** Toàn bộ dữ liệu ảnh và nhãn dạng TSV/JSON sẽ được lưu trực tiếp vào thư mục `/storage/exported` trên **Modal Volume** thay vì lưu ở container tạm thời.
2. **Đồng bộ tự động:** Khi bạn trigger train LayoutLMv3, mô hình sẽ tự động đọc trực tiếp từ thư mục `/storage/exported` để gộp vào tập dữ liệu huấn luyện mà không cần truyền tải thủ công các file ảnh nặng qua internet.

### 7.3. Cách tải các tệp mô hình mới nhất về máy laptop
Khi tiến trình Retrain trên Modal hoàn thành, các tệp weights và mô hình tốt nhất sẽ nằm trên Modal Volume. Bạn có thể kéo chúng về laptop bằng các lệnh:

```bash
# 1. Tải các mô hình NLU (TF-IDF & spaCy NER) đã train về máy
modal volume get expense-ocr-nlu-storage /nlu_models D:\Luan-Van\Project\expense-ocr-nlu\text_nlu\models_backup\

# 2. Tải tập ảnh hóa đơn và nhãn OCR đã export từ Admin về máy
modal volume get expense-ocr-nlu-storage /exported D:\Luan-Van\Project\expense-ocr-nlu\bill_ocr\exported\

# 3. Tải trọng số LayoutLMv3 tốt nhất đã train về máy
modal volume get expense-ocr-nlu-storage /layoutlmv3/model_best.pth D:\Luan-Van\Project\expense-ocr-nlu\bill_ocr\models\layoutlmv3\
```
