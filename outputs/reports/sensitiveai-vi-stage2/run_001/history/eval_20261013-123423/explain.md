<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# reports/<model> - tất cả lần chạy của một model

> **Dùng để làm gì:** Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập.

> **Sinh ra bởi:** `run_experiment` tự cấp số thứ tự tự do kế tiếp.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `run_pointer.txt` | Tên thư mục của lần chạy mới nhất. |
| `explain.md` | File này, kèm bảng liệt kê các lần chạy. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `benchmark.json` | file | 604 B · 2026-10-13 12:01:55 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-13 12:01:27 |
| `classification_report.txt` | file | 389 B · 2026-10-13 12:01:27 |
| `confusion_matrix.png` | file | 55.5 KB · 2026-10-13 12:01:28 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-13 12:01:27 |
| `confusion_matrix_normalized.png` | file | 62.9 KB · 2026-10-13 12:01:29 |
| `confusion_matrix_raw.png` | file | 56.6 KB · 2026-10-13 12:01:28 |
| `evaluation.json` | file | 1.7 KB · 2026-10-13 12:01:56 |
| `metrics.json` | file | 6.5 KB · 2026-10-13 12:01:56 |
| `model_summary.txt` | file | 3.2 KB · 2026-10-13 12:01:56 |
| `per_class_f1.png` | file | 45.8 KB · 2026-10-13 12:01:30 |
| `pr_curve.png` | file | 132.6 KB · 2026-10-13 12:01:29 |
| `pr_curve_data.json` | file | 1.1 MB · 2026-10-13 12:01:29 |
| `roc_curve.png` | file | 127.1 KB · 2026-10-13 12:01:30 |
| `roc_curve_data.json` | file | 250.6 KB · 2026-10-13 12:01:29 |
| `test_predictions.csv` | file | 800.3 KB · 2026-10-13 12:01:55 |
| `thresholds.json` | file | 1.9 KB · 2026-10-13 12:00:47 |
| `thresholds.txt` | file | 318 B · 2026-10-13 12:00:47 |
| `train_loss_curve.png` | file | 77.2 KB · 2026-10-13 12:01:31 |
| `val_accuracy_curve.png` | file | 108.7 KB · 2026-10-13 12:01:30 |
| `val_loss_curve.png` | file | 86.3 KB · 2026-10-13 12:01:30 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-16 16:02:29. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
