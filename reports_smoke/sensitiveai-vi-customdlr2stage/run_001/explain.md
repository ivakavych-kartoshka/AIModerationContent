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
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-04 12:51:21 |
| `class_weights.json` | file | 679 B · 2026-10-04 12:47:10 |
| `classification_report.txt` | file | 389 B · 2026-10-04 12:51:21 |
| `config.yaml` | file | 3.4 KB · 2026-10-04 12:47:10 |
| `confusion_matrix.png` | file | 52.4 KB · 2026-10-04 12:51:21 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-04 12:51:21 |
| `confusion_matrix_normalized.png` | file | 61.6 KB · 2026-10-04 12:51:21 |
| `confusion_matrix_raw.png` | file | 53.7 KB · 2026-10-04 12:51:21 |
| `evaluate.log` | file | 6.8 KB · 2026-10-04 12:51:28 |
| `evaluation.json` | file | 1.1 KB · 2026-10-04 12:51:22 |
| `explain.md` | file | 3.9 KB · 2026-10-04 12:58:19 |
| `metrics.json` | file | 5.6 KB · 2026-10-04 12:51:22 |
| `model_summary.txt` | file | 2.9 KB · 2026-10-04 12:51:22 |
| `per_class_f1.png` | file | 42.3 KB · 2026-10-04 12:51:22 |
| `pr_curve.png` | file | 111.3 KB · 2026-10-04 12:51:22 |
| `pr_curve_data.json` | file | 31.4 KB · 2026-10-04 12:51:21 |
| `roc_curve.png` | file | 106.4 KB · 2026-10-04 12:51:22 |
| `roc_curve_data.json` | file | 14.8 KB · 2026-10-04 12:51:21 |
| `split_manifest_used.json` | file | 423 B · 2026-10-04 12:47:10 |
| `stage_transitions.json` | file | 3.6 KB · 2026-10-04 12:47:35 |
| `test_predictions.csv` | file | 13.6 KB · 2026-10-04 12:51:22 |
| `thresholds.json` | file | 1.5 KB · 2026-10-04 12:51:20 |
| `thresholds.txt` | file | 316 B · 2026-10-04 12:51:20 |
| `train.log` | file | 5.9 KB · 2026-10-04 12:47:42 |
| `train_loss_curve.png` | file | 77.4 KB · 2026-10-04 12:51:22 |
| `training_log.csv` | file | 761 B · 2026-10-04 12:47:34 |
| `training_summary.json` | file | 11.8 KB · 2026-10-04 12:47:36 |
| `val_accuracy_curve.png` | file | 62.4 KB · 2026-10-04 12:51:22 |
| `val_loss_curve.png` | file | 80.0 KB · 2026-10-04 12:51:22 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
