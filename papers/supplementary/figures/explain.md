<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# papers/supplementary/figures - hình của bài phụ lục

> **Dùng để làm gì:** Hình cho bài phụ lục: độ tin cậy (reliability), chồng lấn lỗi giữa các hệ thống, confusion grid, threshold sweep, và bản copy hình so sánh.

> **Sinh ra bởi:** `python scripts/11_supplementary_analyses.py` (+ copy từ `outputs/evaluation/comparison/`)

> **Quy tắc giữ lại:** Sinh lại được; không sửa tay.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `reliability.png` | Biểu đồ độ tin cậy (calibration) theo bin. |
| `error_overlap.png` | Mức chồng lấn lỗi giữa các model. |
| `confusion_grid.png` | Lưới confusion matrix. |
| `threshold_sweep.png` | macro-F1 theo ngưỡng. |
| `comparison/` | Bản copy hình so sánh giữa các model. |
| `per_model/` | Hình riêng từng model (M1..M6). |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `comparison` | thư mục | 16 file, 1.8 MB |
| `per_model` | thư mục | 37 file, 3.2 MB |
| `confusion_grid.png` | file | 156.2 KB · 2026-10-15 21:20:15 |
| `error_overlap.png` | file | 51.3 KB · 2026-10-15 21:20:16 |
| `explain.md` | file | 1.9 KB · 2026-10-19 15:06:20 |
| `reliability.png` | file | 224.0 KB · 2026-10-15 21:20:16 |
| `threshold_sweep.png` | file | 127.3 KB · 2026-10-15 21:20:16 |

## Cách đọc

1. `reliability.png` + `error_overlap.png` là phần bổ sung quan trọng khi phản biện về hiệu chuẩn và mức giống nhau giữa các model.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:39. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
