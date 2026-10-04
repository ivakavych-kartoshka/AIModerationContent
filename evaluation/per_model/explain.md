<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/per_model - số liệu theo từng model

> **Dùng để làm gì:** Gom chỉ số chính của từng model ra một chỗ, tiện để so sánh nhanh mà không phải mở từng `run_XXX`.

> **Sinh ra bởi:** `run_evaluation` (`--output-dir` hoặc mặc định ghi vào đây).

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `<model>/` | Số liệu của một model. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `sensitiveai-vi` | thư mục | 18 file, 537.4 KB |
| `sensitiveai-vi-custom` | thư mục | 18 file, 555.4 KB |
| `sensitiveai-vi-custom-curriculum` | thư mục | 19 file, 642.6 KB |
| `sensitiveai-vi-customdlr2stage` | thư mục | 19 file, 616.1 KB |
| `explain.md` | file | 1.4 KB · 2026-10-04 12:58:19 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:01. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
