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
| `best_model` | thư mục | 8 file, 1.1 GB |
| `last_model` | thư mục | 8 file, 1.1 GB |
| `benchmark.json` | file | 602 B · 2026-10-07 12:43:59 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-07 12:43:34 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-07 07:36:07 |
| `class_weights.json` | file | 731 B · 2026-10-07 07:24:49 |
| `classification_report.txt` | file | 389 B · 2026-10-07 12:43:34 |
| `config.yaml` | file | 4.0 KB · 2026-10-07 07:24:49 |
| `confusion_matrix.png` | file | 56.1 KB · 2026-10-07 12:43:35 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-07 12:43:34 |
| `confusion_matrix_normalized.png` | file | 64.2 KB · 2026-10-07 12:43:35 |
| `confusion_matrix_raw.png` | file | 57.2 KB · 2026-10-07 12:43:35 |
| `evaluate.log` | file | 3.7 KB · 2026-10-07 12:44:00 |
| `evaluation.json` | file | 1.7 KB · 2026-10-07 12:44:00 |
| `explain.md` | file | 3.4 KB · 2026-10-19 15:06:21 |
| `metrics.json` | file | 6.5 KB · 2026-10-07 12:44:00 |
| `model_summary.txt` | file | 3.3 KB · 2026-10-07 12:44:00 |
| `per_class_f1.png` | file | 46.0 KB · 2026-10-07 12:43:36 |
| `pr_curve.png` | file | 133.5 KB · 2026-10-07 12:43:35 |
| `pr_curve_data.json` | file | 418.5 KB · 2026-10-07 12:43:35 |
| `roc_curve.png` | file | 134.7 KB · 2026-10-07 12:43:35 |
| `roc_curve_data.json` | file | 228.7 KB · 2026-10-07 12:43:35 |
| `split_manifest_used.json` | file | 397 B · 2026-10-07 07:24:49 |
| `stage_transitions.json` | file | 6.2 KB · 2026-10-07 11:03:31 |
| `test_predictions.csv` | file | 810.1 KB · 2026-10-07 12:43:59 |
| `thresholds.json` | file | 1.9 KB · 2026-10-07 12:42:55 |
| `thresholds.txt` | file | 315 B · 2026-10-07 12:42:55 |
| `train.log` | file | 3.9 KB · 2026-10-07 11:03:37 |
| `train_loss_curve.png` | file | 115.4 KB · 2026-10-07 12:43:36 |
| `training_log.csv` | file | 4.8 KB · 2026-10-07 11:03:29 |
| `training_summary.json` | file | 21.8 KB · 2026-10-07 11:03:37 |
| `val_accuracy_curve.png` | file | 126.4 KB · 2026-10-07 12:43:36 |
| `val_loss_curve.png` | file | 97.2 KB · 2026-10-07 12:43:36 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
