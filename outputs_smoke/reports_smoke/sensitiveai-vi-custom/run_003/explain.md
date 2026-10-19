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
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 16:35:11 |
| `class_weights.json` | file | 418 B · 2026-10-05 16:15:10 |
| `classification_report.txt` | file | 389 B · 2026-10-05 16:35:11 |
| `config.yaml` | file | 2.4 KB · 2026-10-05 16:15:10 |
| `confusion_matrix.png` | file | 54.8 KB · 2026-10-05 16:35:12 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-05 16:35:11 |
| `confusion_matrix_normalized.png` | file | 61.6 KB · 2026-10-05 16:35:12 |
| `confusion_matrix_raw.png` | file | 55.9 KB · 2026-10-05 16:35:12 |
| `detail_log_terminal.txt` | file | 0 B · 2026-10-05 17:03:10 |
| `evaluate.log` | file | 3.0 KB · 2026-10-05 16:35:13 |
| `evaluation.json` | file | 1.0 KB · 2026-10-05 16:35:13 |
| `explain.md` | file | 3.5 KB · 2026-10-19 15:06:20 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 16:35:13 |
| `model_summary.txt` | file | 2.6 KB · 2026-10-05 16:35:13 |
| `per_class_f1.png` | file | 44.1 KB · 2026-10-05 16:35:12 |
| `pr_curve.png` | file | 132.0 KB · 2026-10-05 16:35:12 |
| `pr_curve_data.json` | file | 1001.3 KB · 2026-10-05 16:35:12 |
| `roc_curve.png` | file | 127.6 KB · 2026-10-05 16:35:12 |
| `roc_curve_data.json` | file | 315.1 KB · 2026-10-05 16:35:12 |
| `split_manifest_used.json` | file | 397 B · 2026-10-05 16:15:10 |
| `stage_transitions.json` | file | 1.5 KB · 2026-10-05 16:31:33 |
| `test_predictions.csv` | file | 795.3 KB · 2026-10-05 16:35:13 |
| `train.log` | file | 3.3 KB · 2026-10-05 16:31:35 |
| `train_loss_curve.png` | file | 45.3 KB · 2026-10-05 16:35:13 |
| `training_log.csv` | file | 594 B · 2026-10-05 16:31:33 |
| `training_summary.json` | file | 6.8 KB · 2026-10-05 16:31:35 |
| `val_accuracy_curve.png` | file | 49.6 KB · 2026-10-05 16:35:12 |
| `val_loss_curve.png` | file | 47.3 KB · 2026-10-05 16:35:13 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
