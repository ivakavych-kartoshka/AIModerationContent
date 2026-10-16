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
| `history` | thư mục | 18 file, 2.5 MB |
| `last_model` | thư mục | 4 file, 1.1 GB |
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 07:15:58 |
| `class_weights.json` | file | 418 B · 2026-10-04 22:58:18 |
| `classification_report.txt` | file | 389 B · 2026-10-05 07:15:58 |
| `config.yaml` | file | 2.2 KB · 2026-10-04 22:58:18 |
| `confusion_matrix.png` | file | 52.2 KB · 2026-10-05 07:15:59 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-05 07:15:58 |
| `confusion_matrix_normalized.png` | file | 59.6 KB · 2026-10-05 07:15:59 |
| `confusion_matrix_raw.png` | file | 53.3 KB · 2026-10-05 07:15:59 |
| `evaluate.log` | file | 19.8 KB · 2026-10-05 07:18:42 |
| `evaluation.json` | file | 1.0 KB · 2026-10-05 07:16:01 |
| `explain.md` | file | 1.6 KB · 2026-10-04 22:58:06 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 07:16:01 |
| `model_summary.txt` | file | 2.6 KB · 2026-10-05 07:16:01 |
| `per_class_f1.png` | file | 42.7 KB · 2026-10-05 07:16:00 |
| `pr_curve.png` | file | 127.3 KB · 2026-10-05 07:15:59 |
| `pr_curve_data.json` | file | 849.0 KB · 2026-10-05 07:15:59 |
| `roc_curve.png` | file | 126.8 KB · 2026-10-05 07:16:00 |
| `roc_curve_data.json` | file | 349.0 KB · 2026-10-05 07:15:59 |
| `split_manifest_used.json` | file | 427 B · 2026-10-04 22:58:18 |
| `stage_transitions.json` | file | 1.5 KB · 2026-10-04 23:15:07 |
| `test_predictions.csv` | file | 791.4 KB · 2026-10-05 07:16:00 |
| `train.log` | file | 14.1 KB · 2026-10-05 01:00:09 |
| `train_loss_curve.png` | file | 42.3 KB · 2026-10-05 07:16:00 |
| `training_log.csv` | file | 588 B · 2026-10-04 23:15:07 |
| `training_summary.json` | file | 6.8 KB · 2026-10-04 23:15:10 |
| `val_accuracy_curve.png` | file | 46.1 KB · 2026-10-05 07:16:00 |
| `val_loss_curve.png` | file | 44.3 KB · 2026-10-05 07:16:00 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-16 16:02:28. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
