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
| `best_model` | thư mục | 5 file, 1.1 GB |
| `history` | thư mục | 23 file, 2.8 MB |
| `last_model` | thư mục | 5 file, 1.1 GB |
| `benchmark.json` | file | 603 B · 2026-10-13 12:35:53 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-13 12:35:28 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-11 22:30:03 |
| `class_weights.json` | file | 577 B · 2026-10-13 14:27:42 |
| `classification_report.txt` | file | 389 B · 2026-10-13 12:35:28 |
| `config.yaml` | file | 2.5 KB · 2026-10-13 14:27:42 |
| `confusion_matrix.png` | file | 55.5 KB · 2026-10-13 12:35:28 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-13 12:35:28 |
| `confusion_matrix_normalized.png` | file | 62.9 KB · 2026-10-13 12:35:29 |
| `confusion_matrix_raw.png` | file | 56.6 KB · 2026-10-13 12:35:28 |
| `evaluate.log` | file | 6.5 KB · 2026-10-13 12:35:53 |
| `evaluation.json` | file | 1.7 KB · 2026-10-13 12:35:53 |
| `explain.md` | file | 3.6 KB · 2026-10-19 15:06:21 |
| `last_training_state.pt` | file | 13.4 KB · 2026-10-13 14:26:55 |
| `last_training_state.pt.tmp` | file | 2.1 GB · 2026-10-13 13:48:44 |
| `metrics.json` | file | 6.5 KB · 2026-10-13 12:35:53 |
| `model_summary.txt` | file | 3.2 KB · 2026-10-13 12:35:53 |
| `per_class_f1.png` | file | 45.8 KB · 2026-10-13 12:35:29 |
| `pr_curve.png` | file | 132.6 KB · 2026-10-13 12:35:29 |
| `pr_curve_data.json` | file | 1.1 MB · 2026-10-13 12:35:29 |
| `roc_curve.png` | file | 127.1 KB · 2026-10-13 12:35:29 |
| `roc_curve_data.json` | file | 250.6 KB · 2026-10-13 12:35:29 |
| `split_manifest_used.json` | file | 397 B · 2026-10-13 14:27:42 |
| `stage_transitions.json` | file | 1.6 KB · 2026-10-13 14:27:45 |
| `test_predictions.csv` | file | 800.3 KB · 2026-10-13 12:35:53 |
| `thresholds.json` | file | 1.9 KB · 2026-10-13 12:34:49 |
| `thresholds.txt` | file | 318 B · 2026-10-13 12:34:49 |
| `train.log` | file | 63.7 KB · 2026-10-13 14:27:42 |
| `train_loss_curve.png` | file | 78.6 KB · 2026-10-13 12:35:29 |
| `training_log.csv` | file | 3.0 KB · 2026-10-13 12:20:44 |
| `training_summary.json` | file | 10.4 KB · 2026-10-13 14:12:13 |
| `val_accuracy_curve.png` | file | 108.0 KB · 2026-10-13 12:35:29 |
| `val_loss_curve.png` | file | 88.1 KB · 2026-10-13 12:35:29 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
