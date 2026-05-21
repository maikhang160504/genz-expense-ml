# Quy tắc gắn nhãn 3 lớp

## 1) Ghi chép
- Dùng cho câu mô tả phát sinh giao dịch, thu nhập, chi tiêu, hoặc nội dung cần đẩy lên Story.
- Thường có số tiền, danh mục chi tiêu, nơi mua, hình thức thanh toán, hoặc mô tả một khoản phát sinh.
- Ví dụ:
  - Ăn sáng 30k
  - Mua trà sữa 45k
  - Lương về 12tr
  - Thu tiền phòng tháng này

## 2) Hành động
- Dùng cho câu ra lệnh thao tác trên dữ liệu tài chính.
- Bao gồm: thống kê, báo cáo, so sánh, tìm kiếm, sửa, xóa, cài đặt hạn mức, đổi thông tin, lọc giao dịch.
- Câu thường là mệnh lệnh hoặc câu hỏi về dữ liệu đã ghi.
- Ví dụ:
  - Tháng này tiêu hết bao nhiêu?
  - Xóa giao dịch vừa rồi
  - Tìm khoản chi nào trên 200k
  - Đặt hạn mức chi tiêu 10tr

## 3) Tán gẫu
- Dùng cho câu xã giao, chào hỏi, hỏi danh tính, khen ngợi, cảm ơn, hỏi ngoài lề.
- Không mang ý định ghi chép hay điều khiển dữ liệu.
- Không liên quan đến tiền bạc hay chi tiêu.
- Nhãn dữ liệu: `intent = Chitchat`, `sentiment = Positive | Neutral | Negative`.
- Ví dụ:
  - Chào bot
  - Bạn là ai?
  - Cảm ơn nha
  - Hôm nay trời đẹp không?
  - Bạn có người yêu chưa?

## 4) Quy tắc ưu tiên khi phân vân
- Nếu câu vừa có xã giao vừa có nội dung thao tác tài chính, ưu tiên Hành động hoặc Ghi chép theo ý chính.
- Nếu câu chỉ là lời chào, cảm ơn, hỏi chuyện vu vơ, gắn Tán gẫu.
- Nếu câu mô tả một khoản tiền hoặc giao dịch, gắn Ghi chép.
- Nếu câu yêu cầu bot làm việc với dữ liệu đã có, gắn Hành động.

## 5) Tóm tắt nhanh
- Ghi chép = nhập dữ liệu mới.
- Hành động = thao tác trên dữ liệu.
- Tán gẫu = nói chuyện xã giao.
