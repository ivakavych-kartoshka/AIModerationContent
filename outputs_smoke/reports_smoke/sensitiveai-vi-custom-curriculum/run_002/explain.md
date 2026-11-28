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
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 07:18:40 |
| `class_weights.json` | file | 731 B · 2026-10-05 00:10:10 |
| `classification_report.txt` | file | 389 B · 2026-10-05 07:18:40 |
| `config.yaml` | file | 4.1 KB · 2026-10-05 00:10:10 |
| `confusion_matrix.png` | file | 55.3 KB · 2026-10-05 07:18:40 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-05 07:18:40 |
| `confusion_matrix_normalized.png` | file | 65.1 KB · 2026-10-05 07:18:41 |
| `confusion_matrix_raw.png` | file | 56.4 KB · 2026-10-05 07:18:41 |
| `evaluate.log` | file | 3.5 KB · 2026-10-05 07:18:42 |
| `evaluation.json` | file | 1.2 KB · 2026-10-05 07:18:42 |
| `explain.md` | file | 3.5 KB · 2026-10-20 07:13:19 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 07:18:41 |
| `model_summary.txt` | file | 3.0 KB · 2026-10-05 07:18:41 |
| `per_class_f1.png` | file | 45.9 KB · 2026-10-05 07:18:41 |
| `pr_curve.png` | file | 131.2 KB · 2026-10-05 07:18:41 |
| `pr_curve_data.json` | file | 834.8 KB · 2026-10-05 07:18:41 |
| `roc_curve.png` | file | 127.3 KB · 2026-10-05 07:18:41 |
| `roc_curve_data.json` | file | 321.8 KB · 2026-10-05 07:18:41 |
| `split_manifest_used.json` | file | 427 B · 2026-10-05 00:10:10 |
| `stage_transitions.json` | file | 5.9 KB · 2026-10-05 01:00:08 |
| `test_predictions.csv` | file | 808.9 KB · 2026-10-05 07:18:41 |
| `thresholds.json` | file | 1.1 KB · 2026-10-05 07:18:07 |
| `thresholds.txt` | file | 316 B · 2026-10-05 07:18:07 |
| `train.log` | file | 3.7 KB · 2026-10-05 01:00:09 |
| `train_loss_curve.png` | file | 72.9 KB · 2026-10-05 07:18:41 |
| `training_log.csv` | file | 1.0 KB · 2026-10-05 01:00:06 |
| `training_summary.json` | file | 17.2 KB · 2026-10-05 01:00:09 |
| `val_accuracy_curve.png` | file | 91.5 KB · 2026-10-05 07:18:41 |
| `val_loss_curve.png` | file | 75.3 KB · 2026-10-05 07:18:41 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
