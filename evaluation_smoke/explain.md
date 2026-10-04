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
| `comparison/` | Bảng và biểu đồ so sánh 4 model. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `comparison` | thư mục | 17 file, 1.1 MB |
| `explain.md` | file | 2.1 KB · 2026-10-04 12:58:19 |

## Cách đọc

1. Muốn xem toàn bộ lịch sử chấm điểm: `evaluation/logs/evaluation_log.csv`.
2. Muốn có bảng để dán vào bài: `evaluation/comparison/model_comparison.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
