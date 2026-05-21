# Yêu cầu sản phẩm (gốc)

1. Nhận dạng sai → user gán lại intent → lưu CSV → train `model_custom` riêng user, ưu tiên nhãn user.
2. Nhận dạng Action → popup xác nhận; đã OK một lần thì lần sau không hỏi lại.
3. Action user xác nhận sai → không thực hiện, báo lỗi, không lưu; log trên server cho admin hoặc user mô tả → AI gán nhãn → train lại model user.

**Luồng triển khai chi tiết (API, DB, merge model, scale):** xem [`ARCHITECTURE_TRIEN_KHAI.md`](ARCHITECTURE_TRIEN_KHAI.md).
