# Báo cáo hoàn thành Kế hoạch 1: Xóa luồng Gemini và kiểm thử Qwen với Modal serve

Báo cáo này tổng hợp quá trình loại bỏ hoàn toàn luồng xử lý Gemini ra khỏi hệ thống spending-diary, chuẩn hóa Qwen2.5 làm mô hình ngôn ngữ lớn duy nhất và kết quả kiểm thử thực tế trên môi trường Modal serve.

## 1. Tóm tắt công việc làm sạch mã nguồn

Hệ thống đã được gỡ bỏ toàn bộ phụ thuộc vào API Gemini, bao gồm cơ chế xoay tua khóa API cũ và các đoạn chuyển tiếp dự phòng. Qwen2.5 triển khai trên Modal và máy cục bộ được chỉ định là nguồn suy luận duy nhất.

| STT | Tệp mã nguồn | Nội dung thực hiện |
| --- | --- | --- |
| 1 | src/llm/gemini_keys.py | Xóa hoàn toàn tệp quản lý khóa API Gemini |
| 2 | src/llm/client.py | Loại bỏ hàm gọi Gemini, chuẩn hóa giao tiếp LLM qua Qwen |
| 3 | src/nlu/llm_intent_handler.py | Xóa các khối thử và lỗi liên quan đến Gemini, chuyển phương thức suy luận qua Qwen Modal |
| 4 | src/nlg/llm_runner.py | Loại bỏ phụ thuộc Gemini trong module sinh câu phản hồi |
| 5 | src/nlu/disambiguation_generator.py | Thay thế cơ chế gọi Gemini bằng LLM Qwen |
| 6 | src/api/app/schemas/nlu.py | Chuẩn hóa các trường dữ liệu đầu ra từ LLM |
| 7 | src/api/app/services/nlu_service.py | Cập nhật logic trích xuất kết quả suy luận sang trường dữ liệu chung |
| 8 | .env và src/api/app/core/config.py | Gỡ bỏ các biến môi trường và cấu hình liên quan đến Gemini |

Bảng 1: Danh sách các tệp mã nguồn được tái cấu trúc trong Kế hoạch 1.
Quá trình làm sạch giúp kiến trúc mã nguồn gọn nhẹ, loại bỏ các đoạn mã dư thừa và đảm bảo hệ thống chỉ tương tác với mô hình Qwen chuyên biệt cho tài chính tiếng Việt.

## 2. Kết quả kiểm thử suy luận với Modal serve

Sau khi hoàn tất chỉnh sửa mã nguồn, tiến trình Modal serve được khởi động để phục vụ endpoint suy luận của Backend FastAPI. Hai mẫu kiểm thử tiêu biểu cho hai luồng xử lý chính được gửi tới hệ thống.

| Mẫu kiểm thử | Câu đầu vào | Ý định nhận diện | Backend xử lý | Câu phản hồi từ Mimo | Kiểm tra Gemini |
| --- | --- | --- | --- | --- | --- |
| Mẫu 1: Ghi chép chi tiêu | Hôm nay đi ăn phở với bạn hết 45k | Record | llm_unified | Hôm nay trời mát mà đi ăn phở với bạn cũng vui nhỉ? Momo đã ghi nhận giao dịch này cho bạn rồi nha! | Không tồn tại |
| Mẫu 2: Hỏi đáp chi tiêu | Tháng này tôi tiêu hết bao nhiêu rồi? | Action | llm_unified | Tháng này cậu tiêu hết bao nhiêu rồi nhỉ? Để Mimo xem giúp nhé! | Không tồn tại |

Bảng 2: Kết quả kiểm thử suy luận hai mẫu câu với dịch vụ Modal serve.
Cả hai mẫu kiểm thử đều đạt mã trạng thái thành công 200, nhận diện chính xác ý định người dùng và sinh câu phản hồi tự nhiên từ Qwen mà không phát sinh bất kỳ tham chiếu nào đến Gemini trong dữ liệu xử lý.

## 3. Quản lý tài nguyên

Ngay sau khi hoàn tất việc gửi mẫu thử và xác nhận dữ liệu đầu ra hợp lệ, tiến trình chạy Modal serve đã được lệnh ngắt kết nối và tắt hoàn toàn nhằm tránh hao phí tài nguyên máy chủ GPU trên cloud.
