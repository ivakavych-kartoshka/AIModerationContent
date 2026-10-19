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
| `benchmark.json` | file | 604 B · 2026-10-11 21:12:03 |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-11 21:11:37 |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-11 18:10:33 |
| `class_weights.json` | file | 581 B · 2026-10-11 17:36:51 |
| `classification_report.txt` | file | 389 B · 2026-10-11 21:11:37 |
| `config.yaml` | file | 3.1 KB · 2026-10-11 17:36:51 |
| `confusion_matrix.png` | file | 58.1 KB · 2026-10-11 21:11:38 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-11 21:11:37 |
| `confusion_matrix_normalized.png` | file | 65.6 KB · 2026-10-11 21:11:38 |
| `confusion_matrix_raw.png` | file | 59.2 KB · 2026-10-11 21:11:38 |
| `evaluate.log` | file | 3.3 KB · 2026-10-11 21:12:04 |
| `evaluation.json` | file | 1.8 KB · 2026-10-11 21:12:03 |
| `explain.md` | file | 3.5 KB · 2026-10-19 15:06:21 |
| `last_training_state.pt` | file | 2.1 GB · 2026-10-11 20:25:24 |
| `metrics.json` | file | 6.5 KB · 2026-10-11 21:12:03 |
| `model_summary.txt` | file | 3.2 KB · 2026-10-11 21:12:03 |
| `per_class_f1.png` | file | 47.5 KB · 2026-10-11 21:11:39 |
| `pr_curve.png` | file | 132.5 KB · 2026-10-11 21:11:38 |
| `pr_curve_data.json` | file | 331.5 KB · 2026-10-11 21:11:38 |
| `roc_curve.png` | file | 136.7 KB · 2026-10-11 21:11:39 |
| `roc_curve_data.json` | file | 182.7 KB · 2026-10-11 21:11:38 |
| `split_manifest_used.json` | file | 397 B · 2026-10-11 17:36:51 |
| `stage_transitions.json` | file | 2.1 KB · 2026-10-11 20:25:25 |
| `test_predictions.csv` | file | 809.3 KB · 2026-10-11 21:12:03 |
| `thresholds.json` | file | 1.1 KB · 2026-10-11 21:10:58 |
| `thresholds.txt` | file | 316 B · 2026-10-11 21:10:58 |
| `train.log` | file | 3.7 KB · 2026-10-11 20:25:28 |
| `train_loss_curve.png` | file | 91.8 KB · 2026-10-11 21:11:39 |
| `training_log.csv` | file | 4.0 KB · 2026-10-11 20:25:21 |
| `training_summary.json` | file | 11.4 KB · 2026-10-11 20:25:28 |
| `val_accuracy_curve.png` | file | 108.9 KB · 2026-10-11 21:11:39 |
| `val_loss_curve.png` | file | 92.4 KB · 2026-10-11 21:11:39 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
