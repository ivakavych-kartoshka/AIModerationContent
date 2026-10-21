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
| `last_model` | thư mục | 5 file, 1.1 GB |
| `benchmark.json` | file | 603 B · 2026-10-13 19:38:12 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-13 19:37:48 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-13 17:07:28 |
| `class_weights.json` | file | 577 B · 2026-10-13 16:21:32 |
| `classification_report.txt` | file | 389 B · 2026-10-13 19:37:48 |
| `config.yaml` | file | 2.5 KB · 2026-10-13 16:21:32 |
| `confusion_matrix.png` | file | 55.3 KB · 2026-10-13 19:37:49 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-13 19:37:48 |
| `confusion_matrix_normalized.png` | file | 62.7 KB · 2026-10-13 19:37:49 |
| `confusion_matrix_raw.png` | file | 56.4 KB · 2026-10-13 19:37:49 |
| `evaluate.log` | file | 3.3 KB · 2026-10-13 19:38:12 |
| `evaluation.json` | file | 1.7 KB · 2026-10-13 19:38:12 |
| `explain.md` | file | 3.5 KB · 2026-10-20 07:13:20 |
| `last_training_state.pt` | file | 2.1 GB · 2026-10-13 19:20:53 |
| `metrics.json` | file | 6.5 KB · 2026-10-13 19:38:12 |
| `model_summary.txt` | file | 3.2 KB · 2026-10-13 19:38:12 |
| `per_class_f1.png` | file | 45.6 KB · 2026-10-13 19:37:50 |
| `pr_curve.png` | file | 133.9 KB · 2026-10-13 19:37:49 |
| `pr_curve_data.json` | file | 1.1 MB · 2026-10-13 19:37:49 |
| `roc_curve.png` | file | 127.0 KB · 2026-10-13 19:37:50 |
| `roc_curve_data.json` | file | 225.1 KB · 2026-10-13 19:37:49 |
| `split_manifest_used.json` | file | 397 B · 2026-10-13 16:21:32 |
| `stage_transitions.json` | file | 2.0 KB · 2026-10-13 19:20:55 |
| `test_predictions.csv` | file | 798.9 KB · 2026-10-13 19:38:12 |
| `thresholds.json` | file | 1.1 KB · 2026-10-13 19:37:10 |
| `thresholds.txt` | file | 316 B · 2026-10-13 19:37:10 |
| `train.log` | file | 3.5 KB · 2026-10-13 19:21:00 |
| `train_loss_curve.png` | file | 79.7 KB · 2026-10-13 19:37:50 |
| `training_log.csv` | file | 3.7 KB · 2026-10-13 19:20:48 |
| `training_summary.json` | file | 9.4 KB · 2026-10-13 19:21:00 |
| `val_accuracy_curve.png` | file | 112.9 KB · 2026-10-13 19:37:50 |
| `val_loss_curve.png` | file | 92.2 KB · 2026-10-13 19:37:50 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:44. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
