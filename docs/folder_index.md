# Chỉ mục thư mục dự án

Sinh tự động bởi `scripts/05_document_folders.py --index`. Mỗi dòng dẫn tới `explain.md` của thư mục đó.

| Folder | Mục đích |
|---|---|
| [`./`](./explain.md) | Thư mục gốc của dự án phân loại ngôn tữ độc hại tiếng Việt (HATE / OFFENSIVE / CLEAN) trên ViHSD, so sánh 4 hệ thống dùng chung backbone ... |
| [`checkpoints/`](checkpoints/explain.md) | Kho tập trung cho các checkpoint muốn giữ lâu dài (không nằm trong một lần chạy cụ thể). Mặc định pipeline không ghi vào đây; nếu bạn sao... |
| [`configs/`](configs/explain.md) | Nguồn sự thật cho mọi tham số thực nghiệm: chia dữ liệu và 4 biến thể mô hình. Mỗi lần train đều lưu bản đã resolve vào `reports/<model>/... |
| [`data/`](data/explain.md) | Ba lớp dữ liệu: bản tải về nguyên bản, bản đã làm sạch, và bản split cố định dùng chung cho mọi model. Chỉ tầng `splits/` là thứ mô hình ... |
| [`docs/`](docs/explain.md) | Tài liệu do script sinh ra, không sửa tay. Hiện có `folder_index.md` - bảng tra cứu mọi thư mục trong dự án. |
| [`evaluation/`](evaluation/explain.md) | Tầng tổng hợp cho 4 model: nhật ký mọi lần chấm điểm, kết quả theo từng model, và bảng/biểu đồ so sánh dùng để viết bài báo. Chi tiết the... |
| [`evaluation_smoke/`](evaluation_smoke/explain.md) | Tầng tổng hợp cho 4 model: nhật ký mọi lần chấm điểm, kết quả theo từng model, và bảng/biểu đồ so sánh dùng để viết bài báo. Chi tiết the... |
| [`experiments/`](experiments/explain.md) | Tách khỏi `reports/` để phân biệt **log/kết quả** với **trọng số mô hình**. Mỗi lần chạy có `experiments/<model>/run_XXX/stage_checkpoint... |
| [`experiments_smoke/`](experiments_smoke/explain.md) | Tách khỏi `reports/` để phân biệt **log/kết quả** với **trọng số mô hình**. Mỗi lần chạy có `experiments/<model>/run_XXX/stage_checkpoint... |
| [`paper/`](paper/explain.md) | Khung bài báo LaTeX. Bảng và hình được **sinh tự động** từ kết quả thực nghiệm, không gõ tay. |
| [`paper_smoke/`](paper_smoke/explain.md) | Khung bài báo LaTeX. Bảng và hình được **sinh tự động** từ kết quả thực nghiệm, không gõ tay. |
| [`reports/`](reports/explain.md) | **Nơi đầu tiên cần tìm khi muốn xem log train.** Mỗi lần chạy một model tạo đúng một thư mục `reports/<model>/run_XXX/`, chứa toàn bộ log... |
| [`reports_smoke/`](reports_smoke/explain.md) | **Nơi đầu tiên cần tìm khi muốn xem log train.** Mỗi lần chạy một model tạo đúng một thư mục `reports/<model>/run_XXX/`, chứa toàn bộ log... |
| [`scripts/`](scripts/explain.md) | 5 script chạy theo thứ tự tăng dần: chuẩn bị dữ liệu -> kiểm tra môi trường -> train -> đánh giá -> so sánh -> sinh tài liệu folder. Mỗi ... |
| [`src/`](src/explain.md) | Toàn bộ logic của dự án, chia theo vai trò. Không có file nào ở đây được sinh ra khi chạy, và không có script nào ghi vào thư mục này. |
| [`src/callbacks/`](src/callbacks/explain.md) | Các bước gắn vào vòng lặp train: ghi CSV từng epoch, lưu best checkpoint, early stopping, điều phối chuyển stage. |
| [`src/common/`](src/common/explain.md) | Tầng nền tảng: ánh xạ nhãn cố định, nạp/merge YAML, quy ước `run_XXX`, seed, logging và registry tài liệu folder. |
| [`src/comparison/`](src/comparison/explain.md) | Đọc kết quả của cả 4 run, dựng bảng tổng hợp, vẽ biểu đồ so sánh, và **sinh tự động** file LaTeX (`model_comparison.tex`, `classification... |
| [`src/data/`](src/data/explain.md) | Tải ViHSD, tiền xử lý, tách split có fingerprint SHA-256, và tạo dataset đã tokenize cho PyTorch. |
| [`src/evaluation/`](src/evaluation/explain.md) | Chấm điểm model đã huấn luyện. Nguyên tắc bất di bất dịch: **chỉ chấm test đúng một lần**, tìm threshold **trên validation**, dùng class ... |
| [`src/losses/`](src/losses/explain.md) | Một nơi duy nhất để dựng loss từ YAML, gồm cross-entropy, focal loss (alpha theo nhãn), class weight và concept-focused curriculum loss. |
| [`src/models/`](src/models/explain.md) | Bọc `microsoft/mdeberta-v3-base` với hai loại head: standard `[CLS]` (model 1) và custom multi-view cls+mean+max (model 2/3/4). Cùng một ... |
| [`src/preprocessing/`](src/preprocessing/explain.md) | Chuẩn hoá văn bản tiếng Việt trước khi mã hoá, theo nguyên tắc **không thao tác nào xóa thông tin được bật mặc định**: mọi tuỳ chọn nguy ... |
| [`src/training/`](src/training/explain.md) | Dựng optimizer theo nhóm DLR, lập kế hoạch stage, chạy vòng lặp train tùy chỉnh (không dùng Hugging Face `Trainer` vì API đổi ở transform... |
| [`reports_smoke/sensitiveai-vi/`](reports_smoke/sensitiveai-vi/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom/`](reports_smoke/sensitiveai-vi-custom/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom-curriculum/`](reports_smoke/sensitiveai-vi-custom-curriculum/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-customdlr2stage/`](reports_smoke/sensitiveai-vi-customdlr2stage/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-customdlr2stage/run_001/`](reports_smoke/sensitiveai-vi-customdlr2stage/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-customdlr2stage/run_001/last_model/`](reports_smoke/sensitiveai-vi-customdlr2stage/run_001/last_model/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-customdlr2stage/run_001/last_model/encoder/`](reports_smoke/sensitiveai-vi-customdlr2stage/run_001/last_model/encoder/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom-curriculum/run_001/`](reports_smoke/sensitiveai-vi-custom-curriculum/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom-curriculum/run_002/`](reports_smoke/sensitiveai-vi-custom-curriculum/run_002/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom-curriculum/run_002/last_model/`](reports_smoke/sensitiveai-vi-custom-curriculum/run_002/last_model/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom-curriculum/run_002/last_model/encoder/`](reports_smoke/sensitiveai-vi-custom-curriculum/run_002/last_model/encoder/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom/run_001/`](reports_smoke/sensitiveai-vi-custom/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom/run_001/history/`](reports_smoke/sensitiveai-vi-custom/run_001/history/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom/run_001/last_model/`](reports_smoke/sensitiveai-vi-custom/run_001/last_model/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom/run_001/last_model/encoder/`](reports_smoke/sensitiveai-vi-custom/run_001/last_model/encoder/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi-custom/run_001/history/eval_20261004-125112/`](reports_smoke/sensitiveai-vi-custom/run_001/history/eval_20261004-125112/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi/run_001/`](reports_smoke/sensitiveai-vi/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi/run_001/history/`](reports_smoke/sensitiveai-vi/run_001/history/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi/run_001/last_model/`](reports_smoke/sensitiveai-vi/run_001/last_model/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi/run_001/history/eval_20261004-124833/`](reports_smoke/sensitiveai-vi/run_001/history/eval_20261004-124833/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports_smoke/sensitiveai-vi/run_001/history/eval_20261004-125103/`](reports_smoke/sensitiveai-vi/run_001/history/eval_20261004-125103/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi/`](reports/sensitiveai-vi/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-custom/`](reports/sensitiveai-vi-custom/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-custom-curriculum/`](reports/sensitiveai-vi-custom-curriculum/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-customdlr2stage/`](reports/sensitiveai-vi-customdlr2stage/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-customdlr2stage/run_001/`](reports/sensitiveai-vi-customdlr2stage/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-customdlr2stage/run_002/`](reports/sensitiveai-vi-customdlr2stage/run_002/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-custom-curriculum/run_001/`](reports/sensitiveai-vi-custom-curriculum/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-custom-curriculum/run_002/`](reports/sensitiveai-vi-custom-curriculum/run_002/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-custom/run_001/`](reports/sensitiveai-vi-custom/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi-custom/run_002/`](reports/sensitiveai-vi-custom/run_002/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi/run_001/`](reports/sensitiveai-vi/run_001/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi/run_002/`](reports/sensitiveai-vi/run_002/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi/run_003/`](reports/sensitiveai-vi/run_003/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`reports/sensitiveai-vi/run_004/`](reports/sensitiveai-vi/run_004/explain.md) | Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập. |
| [`paper_smoke/figures/`](paper_smoke/figures/explain.md) | Bản sao PNG của các biểu đồ so sánh, đặt đúng tên để `main.tex` `\includegraphics` được. |
| [`paper_smoke/tables/`](paper_smoke/tables/explain.md) | Ba bảng cho bài báo, sinh từ kết quả thực nghiệm nên **không bao giờ có sai số nhập tay**. |
| [`paper/figures/`](paper/figures/explain.md) | Bản sao PNG của các biểu đồ so sánh, đặt đúng tên để `main.tex` `\includegraphics` được. |
| [`paper/sections/`](paper/sections/explain.md) | Mỗi mục của bài báo nằm trong một file riêng để `main.tex` gọn và dễ sửa từng phần. |
| [`paper/supplementary/`](paper/supplementary/explain.md) | Chỗ để các tài liệu đi kèm bài báo: bảng chi tiết hơn, log đầy đủ, và thông tin tái lập. |
| [`paper/tables/`](paper/tables/explain.md) | Ba bảng cho bài báo, sinh từ kết quả thực nghiệm nên **không bao giờ có sai số nhập tay**. |
| [`experiments_smoke/sensitiveai-vi/`](experiments_smoke/sensitiveai-vi/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom/`](experiments_smoke/sensitiveai-vi-custom/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/`](experiments_smoke/sensitiveai-vi-custom-curriculum/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/`](experiments_smoke/sensitiveai-vi-customdlr2stage/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/`](experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/`](experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_1/`](experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_1/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_2/`](experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_2/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_2/encoder/`](experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_2/encoder/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_1/encoder/`](experiments_smoke/sensitiveai-vi-customdlr2stage/run_001/stage_checkpoints/stage_1/encoder/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_1/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_1/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_2/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_2/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_3/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_3/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_3/encoder/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_3/encoder/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_2/encoder/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_2/encoder/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_1/encoder/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_002/stage_checkpoints/stage_1/encoder/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/stage_checkpoints/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/stage_checkpoints/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/stage_checkpoints/stage_1/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/stage_checkpoints/stage_1/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/stage_checkpoints/stage_1/encoder/`](experiments_smoke/sensitiveai-vi-custom-curriculum/run_001/stage_checkpoints/stage_1/encoder/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi-custom/run_001/`](experiments_smoke/sensitiveai-vi-custom/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments_smoke/sensitiveai-vi/run_001/`](experiments_smoke/sensitiveai-vi/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi/`](experiments/sensitiveai-vi/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-custom/`](experiments/sensitiveai-vi-custom/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-custom-curriculum/`](experiments/sensitiveai-vi-custom-curriculum/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-customdlr2stage/`](experiments/sensitiveai-vi-customdlr2stage/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-customdlr2stage/run_001/`](experiments/sensitiveai-vi-customdlr2stage/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-customdlr2stage/run_002/`](experiments/sensitiveai-vi-customdlr2stage/run_002/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-custom-curriculum/run_001/`](experiments/sensitiveai-vi-custom-curriculum/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-custom-curriculum/run_002/`](experiments/sensitiveai-vi-custom-curriculum/run_002/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-custom/run_001/`](experiments/sensitiveai-vi-custom/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi-custom/run_002/`](experiments/sensitiveai-vi-custom/run_002/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi/run_001/`](experiments/sensitiveai-vi/run_001/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi/run_002/`](experiments/sensitiveai-vi/run_002/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi/run_003/`](experiments/sensitiveai-vi/run_003/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`experiments/sensitiveai-vi/run_004/`](experiments/sensitiveai-vi/run_004/explain.md) | Một thư mục con `run_XXX` cho mỗi lần chạy. |
| [`evaluation_smoke/comparison/`](evaluation_smoke/comparison/explain.md) | Nơi dễ dễ nhất để đọc kết quả so sánh: một bảng tất cả chỉ số của 4 model (CSV và Markdown), cùng các biểu đồ PR/ROC/validation dùng cho ... |
| [`evaluation/logs/`](evaluation/logs/explain.md) | Mỗi lần chấm test thêm **một dòng** vào CSV (chế độ append, không ghi đè), nên lịch sử đầy đủ được giữ lại. |
| [`evaluation/per_model/`](evaluation/per_model/explain.md) | Gom chỉ số chính của từng model ra một chỗ, tiện để so sánh nhanh mà không phải mở từng `run_XXX`. |
| [`evaluation/per_model/sensitiveai-vi/`](evaluation/per_model/sensitiveai-vi/explain.md) | Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`. |
| [`evaluation/per_model/sensitiveai-vi-custom/`](evaluation/per_model/sensitiveai-vi-custom/explain.md) | Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`. |
| [`evaluation/per_model/sensitiveai-vi-custom-curriculum/`](evaluation/per_model/sensitiveai-vi-custom-curriculum/explain.md) | Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`. |
| [`evaluation/per_model/sensitiveai-vi-customdlr2stage/`](evaluation/per_model/sensitiveai-vi-customdlr2stage/explain.md) | Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`. |
| [`data/processed/`](data/processed/explain.md) | Kết quả làm sạch + remap nhãn canonical. Phục vụ kiểm tra, không phải nơi mô hình đọc. |
| [`data/raw/`](data/raw/explain.md) | Chỗ tải ViHSD về lần đầu. Giữ nguyên để có thể chạy lại tiền xử lý từ đầu. |
| [`data/splits/`](data/splits/explain.md) | **Quan trọng nhất về tính công bằng.** Đây là 3 file CSV mà mọi model đều đọc, kèm SHA-256 của từng file. Mọi lần train/evaluate đều kiểm... |
| [`data/raw/vihsd/`](data/raw/vihsd/explain.md) | File CSV gốc do nhóm UIT cung cấp. Tổng 33.400 dòng (24.048 / 2.672 / 6.680). |
| [`data/processed/vihsd/`](data/processed/vihsd/explain.md) | Cùng số dòng với dữ liệu gốc trừ đúng 2 dòng train có text rỗng (không có gì để mã hoá). Nhãn đã chuyển sang thứ tự dự án: HATE=0, OFFENS... |
