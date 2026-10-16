<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/per_model/<model>

> **Dùng để làm gì:** Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`.

> **Sinh ra bởi:** `run_evaluation`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 07:18:42 |
| `classification_report.txt` | file | 389 B · 2026-10-05 07:18:42 |
| `config.yaml` | file | 4.1 KB · 2026-10-05 07:18:42 |
| `confusion_matrix.png` | file | 55.3 KB · 2026-10-05 07:18:42 |
| `confusion_matrix_normalized.png` | file | 65.1 KB · 2026-10-05 07:18:42 |
| `confusion_matrix_raw.png` | file | 56.4 KB · 2026-10-05 07:18:42 |
| `explain.md` | file | 2.4 KB · 2026-10-17 00:09:27 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 07:18:42 |
| `model_summary.txt` | file | 3.0 KB · 2026-10-05 07:18:42 |
| `pr_curve.png` | file | 131.2 KB · 2026-10-05 07:18:42 |
| `pr_curve_data.json` | file | 834.8 KB · 2026-10-05 07:18:42 |
| `roc_curve.png` | file | 127.3 KB · 2026-10-05 07:18:42 |
| `roc_curve_data.json` | file | 321.8 KB · 2026-10-05 07:18:42 |
| `test_predictions.csv` | file | 808.9 KB · 2026-10-05 07:18:42 |
| `thresholds.json` | file | 1.1 KB · 2026-10-05 07:18:42 |
| `training_log.csv` | file | 1.0 KB · 2026-10-05 07:18:42 |
| `training_summary.json` | file | 17.2 KB · 2026-10-05 07:18:42 |
| `val_accuracy_curve.png` | file | 91.5 KB · 2026-10-05 07:18:42 |
| `val_loss_curve.png` | file | 75.3 KB · 2026-10-05 07:18:42 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
