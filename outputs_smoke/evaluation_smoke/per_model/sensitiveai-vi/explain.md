<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/per_model/<model>

> **Dùng để làm gì:** Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`.

> **Sinh ra bởi:** `run_evaluation`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `Best_model_classification_report.txt` | file | 389 B · 2026-10-05 07:16:01 |
| `classification_report.txt` | file | 389 B · 2026-10-05 07:16:01 |
| `config.yaml` | file | 2.2 KB · 2026-10-05 07:16:01 |
| `confusion_matrix.png` | file | 52.2 KB · 2026-10-05 07:16:01 |
| `confusion_matrix_normalized.png` | file | 59.6 KB · 2026-10-05 07:16:01 |
| `confusion_matrix_raw.png` | file | 53.3 KB · 2026-10-05 07:16:01 |
| `explain.md` | file | 2.3 KB · 2026-10-19 15:06:21 |
| `metrics.json` | file | 6.5 KB · 2026-10-05 07:16:01 |
| `model_summary.txt` | file | 2.6 KB · 2026-10-05 07:16:01 |
| `pr_curve.png` | file | 127.3 KB · 2026-10-05 07:16:01 |
| `pr_curve_data.json` | file | 849.0 KB · 2026-10-05 07:16:01 |
| `roc_curve.png` | file | 126.8 KB · 2026-10-05 07:16:01 |
| `roc_curve_data.json` | file | 349.0 KB · 2026-10-05 07:16:01 |
| `test_predictions.csv` | file | 791.4 KB · 2026-10-05 07:16:01 |
| `training_log.csv` | file | 588 B · 2026-10-05 07:16:01 |
| `training_summary.json` | file | 6.8 KB · 2026-10-05 07:16:01 |
| `val_accuracy_curve.png` | file | 46.1 KB · 2026-10-05 07:16:01 |
| `val_loss_curve.png` | file | 44.3 KB · 2026-10-05 07:16:01 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
