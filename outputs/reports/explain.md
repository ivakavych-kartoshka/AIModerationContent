<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# reports - log và kết quả mỗi lần chạy

> **Dùng để làm gì:** **Nơi đầu tiên cần tìm khi muốn xem log train.** Mỗi lần chạy một model tạo đúng một thư mục `reports/<model>/run_XXX/`, chứa toàn bộ log, artifact và kết quả của riêng lần chạy đó. Các lần chạy không bao giờ ghi đè lẫn nhau.

> **Sinh ra bởi:** `python -m src.training.run_experiment --config <model>` rồi `python -m src.evaluation.run_evaluation --model <model>`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `sensitiveai-vi/` | Model 1 - baseline. |
| `sensitiveai-vi-custom/` | Model 2 - custom head. |
| `sensitiveai-vi-customdlr2stage/` | Model 3 - DLR + 2 stage + focal. |
| `sensitiveai-vi-custom-curriculum/` | Model 4 - curriculum 3 stage. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `sensitiveai-vi` | thư mục | 21 file, 30.5 KB |
| `sensitiveai-vi-custom` | thư mục | 11 file, 17.0 KB |
| `sensitiveai-vi-custom-curriculum` | thư mục | 11 file, 21.6 KB |
| `sensitiveai-vi-customdlr2stage` | thư mục | 11 file, 20.1 KB |
| `explain.md` | file | 2.2 KB · 2026-10-04 12:58:19 |

## Cách đọc

1. Muốn xem log của một lần chạy: `reports/<model>/run_XXX/explain.md`.
2. Muốn biết hiện có bao nhiêu lần chạy và lần nào là mới nhất: bảng **Danh sách các lần chạy** trong chính file này (do `05_document_folders.py` làm mới).
3. Muốn tìm nhanh: dùng `run_pointer.txt` trong `reports/<model>/` (trỏ tới run mới nhất).

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
