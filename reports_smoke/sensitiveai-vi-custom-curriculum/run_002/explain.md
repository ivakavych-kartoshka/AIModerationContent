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
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-04 12:51:26 |
| `class_weights.json` | file | 725 B · 2026-10-04 12:50:12 |
| `classification_report.txt` | file | 389 B · 2026-10-04 12:51:26 |
| `config.yaml` | file | 3.9 KB · 2026-10-04 12:50:12 |
| `confusion_matrix.png` | file | 52.9 KB · 2026-10-04 12:51:26 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-04 12:51:26 |
| `confusion_matrix_normalized.png` | file | 58.4 KB · 2026-10-04 12:51:27 |
| `confusion_matrix_raw.png` | file | 54.2 KB · 2026-10-04 12:51:27 |
| `evaluate.log` | file | 3.4 KB · 2026-10-04 12:51:28 |
| `evaluation.json` | file | 1.1 KB · 2026-10-04 12:51:27 |
| `explain.md` | file | 3.9 KB · 2026-10-04 12:58:19 |
| `metrics.json` | file | 5.4 KB · 2026-10-04 12:51:27 |
| `model_summary.txt` | file | 3.0 KB · 2026-10-04 12:51:27 |
| `per_class_f1.png` | file | 41.7 KB · 2026-10-04 12:51:27 |
| `pr_curve.png` | file | 111.1 KB · 2026-10-04 12:51:27 |
| `pr_curve_data.json` | file | 31.4 KB · 2026-10-04 12:51:27 |
| `roc_curve.png` | file | 107.7 KB · 2026-10-04 12:51:27 |
| `roc_curve_data.json` | file | 14.6 KB · 2026-10-04 12:51:27 |
| `split_manifest_used.json` | file | 423 B · 2026-10-04 12:50:12 |
| `stage_transitions.json` | file | 5.8 KB · 2026-10-04 12:50:48 |
| `test_predictions.csv` | file | 13.7 KB · 2026-10-04 12:51:27 |
| `thresholds.json` | file | 1.1 KB · 2026-10-04 12:51:26 |
| `thresholds.txt` | file | 316 B · 2026-10-04 12:51:26 |
| `train.log` | file | 3.6 KB · 2026-10-04 12:50:49 |
| `train_loss_curve.png` | file | 80.7 KB · 2026-10-04 12:51:27 |
| `training_log.csv` | file | 962 B · 2026-10-04 12:50:47 |
| `training_summary.json` | file | 17.1 KB · 2026-10-04 12:50:49 |
| `val_accuracy_curve.png` | file | 79.3 KB · 2026-10-04 12:51:27 |
| `val_loss_curve.png` | file | 84.8 KB · 2026-10-04 12:51:27 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
