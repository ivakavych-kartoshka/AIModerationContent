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
| `best_model` | thư mục | 4 file, 1.1 GB |
| `last_model` | thư mục | 4 file, 1.1 GB |
| `benchmark.json` | file | 601 B · 2026-10-06 03:55:32 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-06 03:54:33 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-06 00:13:18 |
| `class_weights.json` | file | 418 B · 2026-10-05 22:32:04 |
| `classification_report.txt` | file | 389 B · 2026-10-06 03:54:33 |
| `config.yaml` | file | 2.2 KB · 2026-10-05 22:32:04 |
| `confusion_matrix.png` | file | 54.6 KB · 2026-10-06 03:54:34 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-06 03:54:33 |
| `confusion_matrix_normalized.png` | file | 60.9 KB · 2026-10-06 03:54:35 |
| `confusion_matrix_raw.png` | file | 55.6 KB · 2026-10-06 03:54:34 |
| `evaluate.log` | file | 3.2 KB · 2026-10-06 03:55:32 |
| `evaluation.json` | file | 1.6 KB · 2026-10-06 03:55:32 |
| `explain.md` | file | 1.4 KB · 2026-10-05 22:31:49 |
| `metrics.json` | file | 6.5 KB · 2026-10-06 03:55:32 |
| `model_summary.txt` | file | 2.9 KB · 2026-10-06 03:55:32 |
| `per_class_f1.png` | file | 43.0 KB · 2026-10-06 03:54:35 |
| `pr_curve.png` | file | 131.9 KB · 2026-10-06 03:54:35 |
| `pr_curve_data.json` | file | 905.0 KB · 2026-10-06 03:54:35 |
| `roc_curve.png` | file | 127.1 KB · 2026-10-06 03:54:35 |
| `roc_curve_data.json` | file | 333.9 KB · 2026-10-06 03:54:35 |
| `split_manifest_used.json` | file | 397 B · 2026-10-05 22:32:04 |
| `stage_transitions.json` | file | 1.5 KB · 2026-10-06 01:22:09 |
| `test_predictions.csv` | file | 807.1 KB · 2026-10-06 03:55:32 |
| `train.log` | file | 3.4 KB · 2026-10-06 01:22:14 |
| `train_loss_curve.png` | file | 77.5 KB · 2026-10-06 03:54:35 |
| `training_log.csv` | file | 3.6 KB · 2026-10-06 01:22:09 |
| `training_summary.json` | file | 10.0 KB · 2026-10-06 01:22:14 |
| `val_accuracy_curve.png` | file | 111.8 KB · 2026-10-06 03:54:35 |
| `val_loss_curve.png` | file | 85.7 KB · 2026-10-06 03:54:35 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-16 16:02:29. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
