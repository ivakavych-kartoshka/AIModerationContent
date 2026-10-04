<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# comparison - so sánh 4 model

> **Dùng để làm gì:** Đọc kết quả của cả 4 run, dựng bảng tổng hợp, vẽ biểu đồ so sánh, và **sinh tự động** file LaTeX (`model_comparison.tex`, `classification_results.tex`, `ablation_results.tex`) cho bài báo.

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Kết quả nằm ở `evaluation/comparison/` và `paper/tables/`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `compare_models.py` | CLI so sánh. Bảng ablation sinh từ đúng 4 hệ thống nên trả lời trực tiếp: đổi head được bao nhiêu, thêm DLR/focal/2 stage được bao nhiêu, curriculum 3 stage hơn 2 stage bao nhiêu. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 2 file, 42.7 KB |
| `__init__.py` | file | 199 B · 2026-10-04 10:44:49 |
| `compare_models.py` | file | 26.8 KB · 2026-10-04 12:55:08 |
| `explain.md` | file | 1.7 KB · 2026-10-04 12:58:19 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
