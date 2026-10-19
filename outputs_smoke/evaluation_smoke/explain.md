<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation - kết quả đánh giá tổng hợp

> **Dùng để làm gì:** Tầng tổng hợp cho 4 model: nhật ký mọi lần chấm điểm, kết quả theo từng model, và bảng/biểu đồ so sánh dùng để viết bài báo. Chi tiết theo từng run vẫn nằm trong `reports/<model>/run_XXX/`.

> **Sinh ra bởi:** `python -m src.evaluation.run_evaluation ...` và `python scripts/03_compare_models.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `logs/` | Nhật ký CSV của **mọi** lần chấm điểm - nối các lần chạy lại thành một dòng thời gian. |
| `per_model/` | Bản sao số liệu chính của từng model, gom lại một chỗ. |
| `comparison/` | Bảng và biểu đồ so sánh 4 model + so sánh 2 rule engine. |
| `analysis/` | Ba báo cáo phân tích hậu nghiệm: chống trùng lặp giữa các split, kiểm định thống kê 4 model, supplementary (calibration/MCC). |
| `rule_engine/` | Kết quả đánh giá rule engine từ vựng độc lập (plan §39, §44). |
| `rule_engine_tuned/` | Rule engine với cấu hình đã tune (`configs/rule_engine_tuned.yaml`) - bản tốt nhất, kèm hồ sơ tune (`tuning_trials.csv`, `tuning_report.json`). |
| `fusion/` | Ghép điểm rule engine với xác suất của từng model AI (plan §40-41). |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `comparison` | thư mục | 17 file, 1.3 MB |
| `logs` | thư mục | 2 file, 4.1 KB |
| `per_model` | thư mục | 75 file, 10.3 MB |
| `rule_engine` | thư mục | 3 file, 8.0 KB |
| `explain.md` | file | 2.8 KB · 2026-10-19 15:06:20 |

## Cách đọc

1. Muốn xem toàn bộ lịch sử chấm điểm: `evaluation/logs/evaluation_log.csv`.
2. Muốn có bảng để dán vào bài: `evaluation/comparison/model_comparison.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
