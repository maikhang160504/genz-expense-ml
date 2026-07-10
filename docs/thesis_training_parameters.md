# Tài liệu Thông số Huấn luyện Mô hình (Training Parameters)

Tài liệu này tổng hợp toàn bộ các thông số huấn luyện (hyperparameters), cấu hình tối ưu hóa và môi trường chạy của hai mô hình chính trong đề tài Luận văn: **PhoGPT-4B** (NLU) và **LayoutLMv3** (KIE/OCR).

---

## 1. Mô hình Phân tích Ý định & Thực thể (NLU) - PhoGPT-4B

Mô hình PhoGPT-4B (`vinai/PhoGPT-4B`) được tinh chỉnh (fine-tune) theo dạng sinh văn bản (Causal LM) để phân tích ý định người dùng thành định dạng JSON có chứa `intent`, `action_type`, `slots`, `emotion` và câu phản hồi.

### Chi tiết Hyperparameters
| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Model Base** | `vinai/PhoGPT-4B` | Mô hình ngôn ngữ lớn tối ưu cho tiếng Việt |
| **Max Sequence Length** | `8192` | Độ dài ngữ cảnh tối đa nhận vào mô hình |
| **Precision** | `amp_bf16` | Bfloat16 Mixed Precision tự động |
| **Optimizer** | `Decoupled LionW` | Thuật toán tối ưu Lion tách biệt weight decay |
| **Learning Rate (LR)** | `5e-5` | Tốc độ học tập |
| **Weight Decay** | `1e-7` | Tham số phạt L2 tránh quá khớp |
| **Global Batch Size** | `64` | Tổng kích thước batch trên mỗi bước cập nhật trọng số |
| **Microbatch Size** | `1` | Kích thước batch thực tế trên mỗi GPU per step |
| **Max Duration** | `3ep` (3 Epochs) | Số chu kỳ huấn luyện tối đa |
| **Learning Rate Scheduler**| `cosine_with_warmup` | Điều chỉnh giảm LR theo hàm Cosine |
| **Warmup Steps** | `200ba` | Số bước khởi động tăng dần tốc độ học |
| **Gradient Clipping** | `norm` (threshold = 1.0) | Cắt bớt gradient tránh bùng nổ gradient |

### Phân tán & Tiết kiệm Bộ nhớ (Distributed/FSDP)
* **Strategy**: `FULL_SHARD` (Fully Sharded Data Parallel) - Phân mảnh hoàn toàn tham số mô hình, gradient và trạng thái optimizer trên các GPU.
* **Mixed Precision**: `PURE` - Chạy trực tiếp trên độ chính xác thấp để giảm dung lượng VRAM.
* **Activation Checkpointing**: `false` (do H100 GPU có đủ VRAM lớn).

---

## 2. Mô hình Trích xuất Thông tin Hóa đơn (KIE) - LayoutLMv3

Mô hình LayoutLMv3 (`microsoft/layoutlmv3-base`) được tinh chỉnh để thực hiện phân loại token (Token Classification) trên dữ liệu văn bản kết hợp tọa độ ảnh (bounding boxes) thu được từ hóa đơn.

### Chi tiết Hyperparameters
| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Model Base** | `microsoft/layoutlmv3-base` | Mô hình đa phương thức (Multimodal Transformer) |
| **Max Sequence Length** | `512` | Độ dài chuỗi token tối đa sau khi phân tách từ hóa đơn |
| **Optimizer** | `AdamW` | Thuật toán tối ưu Adam kèm Weight Decay |
| **Learning Rate (LR)** | `5e-5` | Tốc độ học tập |
| **Batch Size per Device**| `4` | Kích thước batch trên mỗi thiết bị GPU |
| **Training Epochs** | Tối đa `15` | Số lượng epoch huấn luyện tối đa |
| **Early Stopping** | `patience = 3` | Dừng sớm nếu chỉ số F1 trên tập validation không tăng sau 3 epoch |
| **Evaluation Strategy** | `epoch` | Đánh giá mô hình sau mỗi epoch |
| **Save Strategy** | `epoch` | Lưu checkpoint sau mỗi epoch |
| **Metric for Best Model**| `f1` | Lựa chọn checkpoint tốt nhất dựa trên điểm F1 |
| **Label mapping** | `SELLER` (15), `ADDRESS` (16), `TIMESTAMP` (17), `TOTAL_COST` (18) | Các lớp thực thể hóa đơn cần trích xuất |

---

## 3. Cấu hình Môi trường Phần cứng & Nền tảng (Modal Cloud GPU)

Quá trình huấn luyện được thực hiện thông qua nền tảng Serverless GPU **Modal**:
* **PhoGPT-4B (NLU)**: Chạy trên GPU **Nvidia H100 (80GB VRAM)**. Sử dụng thư viện `accelerate` kết hợp `PyTorch FSDP` giúp tăng tốc độ train lên tối đa, rút ngắn thời gian tinh chỉnh còn vài phút với tập dữ liệu tối ưu hóa.
* **LayoutLMv3 (KIE)**: Chạy trên GPU **Nvidia A10G (24GB VRAM)** hoặc **L4 (24GB VRAM)** sử dụng thư viện `Hugging Face Trainer`.
