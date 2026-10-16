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
| `benchmark.json` | file | 604 B · 2026-10-07 07:17:33 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-07 07:17:08 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-07 02:40:43 |
| `class_weights.json` | file | 682 B · 2026-10-06 23:51:40 |
| `classification_report.txt` | file | 389 B · 2026-10-07 07:17:08 |
| `config.yaml` | file | 3.5 KB · 2026-10-06 23:51:40 |
| `confusion_matrix.png` | file | 58.7 KB · 2026-10-07 07:17:08 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-07 07:17:08 |
| `confusion_matrix_normalized.png` | file | 64.2 KB · 2026-10-07 07:17:09 |
| `confusion_matrix_raw.png` | file | 59.7 KB · 2026-10-07 07:17:08 |
| `evaluate.log` | file | 3.6 KB · 2026-10-07 07:17:34 |
| `evaluation.json` | file | 1.7 KB · 2026-10-07 07:17:34 |
| `explain.md` | file | 3.4 KB · 2026-10-17 00:09:28 |
| `metrics.json` | file | 6.5 KB · 2026-10-07 07:17:33 |
| `model_summary.txt` | file | 3.2 KB · 2026-10-07 07:17:33 |
| `per_class_f1.png` | file | 46.5 KB · 2026-10-07 07:17:09 |
| `pr_curve.png` | file | 131.4 KB · 2026-10-07 07:17:09 |
| `pr_curve_data.json` | file | 375.7 KB · 2026-10-07 07:17:09 |
| `roc_curve.png` | file | 135.4 KB · 2026-10-07 07:17:09 |
| `roc_curve_data.json` | file | 203.7 KB · 2026-10-07 07:17:09 |
| `split_manifest_used.json` | file | 397 B · 2026-10-06 23:51:40 |
| `stage_transitions.json` | file | 3.8 KB · 2026-10-07 03:48:37 |
| `test_predictions.csv` | file | 810.4 KB · 2026-10-07 07:17:33 |
| `thresholds.json` | file | 1.1 KB · 2026-10-07 07:16:29 |
| `thresholds.txt` | file | 316 B · 2026-10-07 07:16:29 |
| `train.log` | file | 3.7 KB · 2026-10-07 03:48:42 |
| `train_loss_curve.png` | file | 106.5 KB · 2026-10-07 07:17:09 |
| `training_log.csv` | file | 5.3 KB · 2026-10-07 03:48:35 |
| `training_summary.json` | file | 16.6 KB · 2026-10-07 03:48:42 |
| `val_accuracy_curve.png` | file | 125.7 KB · 2026-10-07 07:17:09 |
| `val_loss_curve.png` | file | 103.8 KB · 2026-10-07 07:17:09 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:21. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
