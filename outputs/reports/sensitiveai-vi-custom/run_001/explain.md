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
| `benchmark.json` | file | 604 B · 2026-10-06 08:44:24 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-06 08:43:59 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-06 06:35:01 |
| `class_weights.json` | file | 418 B · 2026-10-06 03:52:43 |
| `classification_report.txt` | file | 389 B · 2026-10-06 08:43:59 |
| `config.yaml` | file | 2.4 KB · 2026-10-06 03:52:43 |
| `confusion_matrix.png` | file | 54.8 KB · 2026-10-06 08:44:00 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-06 08:43:59 |
| `confusion_matrix_normalized.png` | file | 62.9 KB · 2026-10-06 08:44:00 |
| `confusion_matrix_raw.png` | file | 55.9 KB · 2026-10-06 08:44:00 |
| `evaluate.log` | file | 3.2 KB · 2026-10-06 08:44:25 |
| `evaluation.json` | file | 1.6 KB · 2026-10-06 08:44:25 |
| `explain.md` | file | 3.3 KB · 2026-10-19 15:06:21 |
| `metrics.json` | file | 6.4 KB · 2026-10-06 08:44:25 |
| `model_summary.txt` | file | 2.9 KB · 2026-10-06 08:44:25 |
| `per_class_f1.png` | file | 44.8 KB · 2026-10-06 08:44:00 |
| `pr_curve.png` | file | 133.7 KB · 2026-10-06 08:44:00 |
| `pr_curve_data.json` | file | 624.3 KB · 2026-10-06 08:44:00 |
| `roc_curve.png` | file | 129.1 KB · 2026-10-06 08:44:00 |
| `roc_curve_data.json` | file | 274.5 KB · 2026-10-06 08:44:00 |
| `split_manifest_used.json` | file | 397 B · 2026-10-06 03:52:43 |
| `stage_transitions.json` | file | 1.5 KB · 2026-10-06 07:43:40 |
| `test_predictions.csv` | file | 808.6 KB · 2026-10-06 08:44:24 |
| `train.log` | file | 3.5 KB · 2026-10-06 07:43:46 |
| `train_loss_curve.png` | file | 82.0 KB · 2026-10-06 08:44:01 |
| `training_log.csv` | file | 4.9 KB · 2026-10-06 07:43:40 |
| `training_summary.json` | file | 11.2 KB · 2026-10-06 07:43:46 |
| `val_accuracy_curve.png` | file | 118.1 KB · 2026-10-06 08:44:01 |
| `val_loss_curve.png` | file | 91.0 KB · 2026-10-06 08:44:01 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
