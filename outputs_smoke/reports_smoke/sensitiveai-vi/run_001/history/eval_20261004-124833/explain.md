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
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-04 12:48:16 |
| `classification_report.txt` | file | 389 B · 2026-10-04 12:48:16 |
| `confusion_matrix.png` | file | 49.8 KB · 2026-10-04 12:48:17 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-04 12:48:16 |
| `confusion_matrix_normalized.png` | file | 55.4 KB · 2026-10-04 12:48:17 |
| `confusion_matrix_raw.png` | file | 51.1 KB · 2026-10-04 12:48:17 |
| `evaluation.json` | file | 1008 B · 2026-10-04 12:48:18 |
| `explain.md` | file | 3.2 KB · 2026-10-04 12:58:19 |
| `metrics.json` | file | 5.4 KB · 2026-10-04 12:48:18 |
| `model_summary.txt` | file | 2.6 KB · 2026-10-04 12:48:18 |
| `per_class_f1.png` | file | 38.7 KB · 2026-10-04 12:48:17 |
| `pr_curve.png` | file | 100.4 KB · 2026-10-04 12:48:17 |
| `pr_curve_data.json` | file | 30.7 KB · 2026-10-04 12:48:17 |
| `roc_curve.png` | file | 104.3 KB · 2026-10-04 12:48:17 |
| `roc_curve_data.json` | file | 16.6 KB · 2026-10-04 12:48:17 |
| `test_predictions.csv` | file | 13.6 KB · 2026-10-04 12:48:18 |
| `train_loss_curve.png` | file | 50.1 KB · 2026-10-04 12:48:18 |
| `val_accuracy_curve.png` | file | 46.7 KB · 2026-10-04 12:48:17 |
| `val_loss_curve.png` | file | 48.6 KB · 2026-10-04 12:48:17 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
