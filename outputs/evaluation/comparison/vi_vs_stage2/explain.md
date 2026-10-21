<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/comparison/vi_vs_stage2 - so hai run cùng model, khác cấu hình

> **Dùng để làm gì:** So sánh riêng cặp `sensitiveai-vi` (run 1 tầng) với `sensitiveai-vi-stage2` (2 tầng): bảng metric trên validation + các hình PR/ROC/learning curve của **mọi** model để làm phụ lục.

> **Sinh ra bởi:** `python scripts/03_compare_models.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `model_comparison.csv / .md` | Bảng metric validation của hai run được so (accuracy, macro-F1, per-class F1). |
| `comparison_summary.json` | Cùng số liệu ở dạng máy đọc. |
| `*_all_models.png` | Hình so mọi model: accuracy, macro-F1, per-class F1, PR/ROC theo từng nhãn, validation loss/accuracy. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `accuracy_all_models.png` | file | 51.5 KB · 2026-10-13 12:36:16 |
| `comparison_summary.json` | file | 4.3 KB · 2026-10-13 12:36:16 |
| `explain.md` | file | 2.9 KB · 2026-10-20 07:13:20 |
| `macro_f1_all_models.png` | file | 47.0 KB · 2026-10-13 12:36:16 |
| `model_comparison.csv` | file | 1.4 KB · 2026-10-13 12:36:15 |
| `model_comparison.md` | file | 2.9 KB · 2026-10-13 12:36:15 |
| `per_class_f1_all_models.png` | file | 49.3 KB · 2026-10-13 12:36:16 |
| `pr_clean_all_models.png` | file | 65.3 KB · 2026-10-13 12:36:16 |
| `pr_curve_all_models.png` | file | 91.8 KB · 2026-10-13 12:36:15 |
| `pr_hate_all_models.png` | file | 93.4 KB · 2026-10-13 12:36:15 |
| `pr_offensive_all_models.png` | file | 90.6 KB · 2026-10-13 12:36:16 |
| `roc_clean_all_models.png` | file | 90.4 KB · 2026-10-13 12:36:16 |
| `roc_curve_all_models.png` | file | 99.8 KB · 2026-10-13 12:36:15 |
| `roc_hate_all_models.png` | file | 89.1 KB · 2026-10-13 12:36:15 |
| `roc_offensive_all_models.png` | file | 89.9 KB · 2026-10-13 12:36:16 |
| `validation_accuracy_all_models.png` | file | 131.4 KB · 2026-10-13 12:36:16 |
| `validation_loss_all_models.png` | file | 88.2 KB · 2026-10-13 12:36:16 |

## Cách đọc

1. Dùng khi cần trả lời '2 tầng có hơn 1 tầng không?': so cột macro-F1 giữa hai run trong `model_comparison.md`.
2. Đây là số **validation**, không phải test.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:45. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
