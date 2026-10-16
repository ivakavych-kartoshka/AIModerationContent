<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# reports/<model> - tất cả lần chạy của một model

> **Dùng để làm gì:** Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập.

> **Sinh ra bởi:** `run_experiment` tự cấp số thứ tự tự do kế tiếp.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `run_pointer.txt` | Tên thư mục của lần chạy mới nhất. |
| `explain.md` | File này, kèm bảng liệt kê các lần chạy. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `last_model` | thư mục | 8 file, 1.1 GB |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 07:17:47 |
| `class_weights.json` | file | 682 B · 2026-10-04 23:32:20 |
| `classification_report.txt` | file | 389 B · 2026-10-05 07:17:47 |
| `config.yaml` | file | 3.5 KB · 2026-10-04 23:32:20 |
| `confusion_matrix.png` | file | 56.9 KB · 2026-10-05 07:17:47 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-05 07:17:47 |
| `confusion_matrix_normalized.png` | file | 65.0 KB · 2026-10-05 07:17:47 |
| `confusion_matrix_raw.png` | file | 57.9 KB · 2026-10-05 07:17:47 |
| `evaluate.log` | file | 7.6 KB · 2026-10-05 07:18:42 |
| `evaluation.json` | file | 1.2 KB · 2026-10-05 07:17:48 |
| `explain.md` | file | 3.5 KB · 2026-10-17 00:09:27 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 07:17:48 |
| `model_summary.txt` | file | 2.9 KB · 2026-10-05 07:17:48 |
| `per_class_f1.png` | file | 47.3 KB · 2026-10-05 07:17:48 |
| `pr_curve.png` | file | 131.9 KB · 2026-10-05 07:17:47 |
| `pr_curve_data.json` | file | 1.1 MB · 2026-10-05 07:17:47 |
| `roc_curve.png` | file | 127.6 KB · 2026-10-05 07:17:47 |
| `roc_curve_data.json` | file | 214.4 KB · 2026-10-05 07:17:47 |
| `split_manifest_used.json` | file | 427 B · 2026-10-04 23:32:20 |
| `stage_transitions.json` | file | 3.6 KB · 2026-10-05 00:10:01 |
| `test_predictions.csv` | file | 787.1 KB · 2026-10-05 07:17:48 |
| `thresholds.json` | file | 1.9 KB · 2026-10-05 07:17:13 |
| `thresholds.txt` | file | 318 B · 2026-10-05 07:17:13 |
| `train.log` | file | 7.3 KB · 2026-10-05 01:00:09 |
| `train_loss_curve.png` | file | 80.1 KB · 2026-10-05 07:17:48 |
| `training_log.csv` | file | 830 B · 2026-10-05 00:10:00 |
| `training_summary.json` | file | 11.9 KB · 2026-10-05 00:10:02 |
| `val_accuracy_curve.png` | file | 95.3 KB · 2026-10-05 07:17:48 |
| `val_loss_curve.png` | file | 83.7 KB · 2026-10-05 07:17:48 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
