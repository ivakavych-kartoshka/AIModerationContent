<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation - kết quả đánh giá tổng hợp

> **Dùng để làm gì:** Tầng tổng hợp cho 4 model: nhật ký mọi lần chấm điểm, kết quả theo từng model, và bảng/biểu đồ so sánh dùng để viết bài báo. Chi tiết theo từng run vẫn nằm trong `reports/<model>/run_XXX/`.

> **Sinh ra bởi:** `python -m src.evaluation.run_evaluation ...` và `python scripts/03_compare_models.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `logs/` | Nhật ký CSV của **mọi** lần chấm điểm - nối các lần chạy lại thành một dòng thời gian. |
| `per_model/` | Bản sao số liệu chính của từng model, gom lại một chỗ. |
| `comparison/` | Bảng và biểu đồ so sánh 4 model. |
| `rule_engine/` | Kết quả đánh giá rule engine từ vựng độc lập (plan §39, §44). |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `analysis` | thư mục | 7 file, 73.4 KB |
| `comparison` | thư mục | 36 file, 2.9 MB |
| `logs` | thư mục | 2 file, 3.7 KB |
| `per_model` | thư mục | 119 file, 13.8 MB |
| `rule_engine` | thư mục | 11 file, 116.7 KB |
| `explain.md` | file | 2.1 KB · 2026-10-17 00:09:27 |

## Cách đọc

1. Muốn xem toàn bộ lịch sử chấm điểm: `evaluation/logs/evaluation_log.csv`.
2. Muốn có bảng để dán vào bài: `evaluation/comparison/model_comparison.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
