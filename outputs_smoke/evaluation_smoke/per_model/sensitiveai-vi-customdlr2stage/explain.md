<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/per_model/<model>

> **Dùng để làm gì:** Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`.

> **Sinh ra bởi:** `run_evaluation`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 07:17:48 |
| `classification_report.txt` | file | 389 B · 2026-10-05 07:17:48 |
| `config.yaml` | file | 3.5 KB · 2026-10-05 07:17:48 |
| `confusion_matrix.png` | file | 56.9 KB · 2026-10-05 07:17:48 |
| `confusion_matrix_normalized.png` | file | 65.0 KB · 2026-10-05 07:17:48 |
| `confusion_matrix_raw.png` | file | 57.9 KB · 2026-10-05 07:17:48 |
| `explain.md` | file | 2.4 KB · 2026-10-17 00:09:27 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 07:17:48 |
| `model_summary.txt` | file | 2.9 KB · 2026-10-05 07:17:48 |
| `pr_curve.png` | file | 131.9 KB · 2026-10-05 07:17:48 |
| `pr_curve_data.json` | file | 1.1 MB · 2026-10-05 07:17:48 |
| `roc_curve.png` | file | 127.6 KB · 2026-10-05 07:17:48 |
| `roc_curve_data.json` | file | 214.4 KB · 2026-10-05 07:17:48 |
| `test_predictions.csv` | file | 787.1 KB · 2026-10-05 07:17:48 |
| `thresholds.json` | file | 1.9 KB · 2026-10-05 07:17:48 |
| `training_log.csv` | file | 830 B · 2026-10-05 07:17:48 |
| `training_summary.json` | file | 11.9 KB · 2026-10-05 07:17:48 |
| `val_accuracy_curve.png` | file | 95.3 KB · 2026-10-05 07:17:48 |
| `val_loss_curve.png` | file | 83.7 KB · 2026-10-05 07:17:48 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
