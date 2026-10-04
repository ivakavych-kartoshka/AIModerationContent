<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/comparison - bảng và biểu đồ so sánh

> **Dùng để làm gì:** Nơi dễ dễ nhất để đọc kết quả so sánh: một bảng tất cả chỉ số của 4 model (CSV và Markdown), cùng các biểu đồ PR/ROC/validation dùng cho bài báo.

> **Sinh ra bởi:** `python scripts/03_compare_models.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `model_comparison.csv` | Bảng tổng hợp tất cả chỉ số của 4 model. |
| `model_comparison.md` | Cùng bảng dạng Markdown, dễ đọc và dán vào bài báo. |
| `comparison_summary.json` | Tóm tắt dạng máy-đọc. |
| `pr_curve_all_models.png` | PR curve macro của 4 model trên cùng một trục. |
| `roc_curve_all_models.png` | ROC curve macro của 4 model. |
| `pr_hate_all_models.png` | PR curve riêng cho lớp HATE (lớp hiếm, khó nhất). |
| `pr_offensive_all_models.png` | PR curve riêng cho lớp OFFENSIVE. |
| `pr_clean_all_models.png` | PR curve riêng cho lớp CLEAN. |
| `roc_hate_all_models.png` | ROC curve riêng cho lớp HATE. |
| `roc_offensive_all_models.png` | ROC curve riêng cho lớp OFFENSIVE. |
| `roc_clean_all_models.png` | ROC curve riêng cho lớp CLEAN. |
| `validation_accuracy_all_models.png` | Đường cong validation accuracy của 4 model theo epoch. |
| `validation_loss_all_models.png` | Đường cong validation loss của 4 model theo epoch. |
| `macro_f1_all_models.png` | So sánh macro F1 dạng thanh. |
| `accuracy_all_models.png` | So sánh accuracy dạng thanh (tham khảo, vì dữ liệu mất cân bằng). |
| `per_class_f1_all_models.png` | F1 theo từng lớp của 4 model. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `accuracy_all_models.png` | file | 65.2 KB · 2026-10-04 12:55:20 |
| `comparison_summary.json` | file | 4.3 KB · 2026-10-04 12:55:20 |
| `explain.md` | file | 4.1 KB · 2026-10-04 12:58:19 |
| `macro_f1_all_models.png` | file | 62.4 KB · 2026-10-04 12:55:20 |
| `model_comparison.csv` | file | 1.8 KB · 2026-10-04 12:55:19 |
| `model_comparison.md` | file | 4.4 KB · 2026-10-04 12:55:19 |
| `per_class_f1_all_models.png` | file | 58.5 KB · 2026-10-04 12:55:20 |
| `pr_clean_all_models.png` | file | 93.4 KB · 2026-10-04 12:55:19 |
| `pr_curve_all_models.png` | file | 118.4 KB · 2026-10-04 12:55:19 |
| `pr_hate_all_models.png` | file | 98.1 KB · 2026-10-04 12:55:19 |
| `pr_offensive_all_models.png` | file | 84.6 KB · 2026-10-04 12:55:19 |
| `roc_clean_all_models.png` | file | 90.3 KB · 2026-10-04 12:55:20 |
| `roc_curve_all_models.png` | file | 125.8 KB · 2026-10-04 12:55:19 |
| `roc_hate_all_models.png` | file | 86.6 KB · 2026-10-04 12:55:19 |
| `roc_offensive_all_models.png` | file | 86.8 KB · 2026-10-04 12:55:19 |
| `validation_accuracy_all_models.png` | file | 90.9 KB · 2026-10-04 12:55:20 |
| `validation_loss_all_models.png` | file | 101.3 KB · 2026-10-04 12:55:20 |

## Cách đọc

1. Bảng quan trọng nhất: `model_comparison.md` (macro F1, weighted F1, per-class F1, latency).
2. Giải thích vì sao accuracy không phải số liệu chính: xem mục 2.2 của `README.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:01. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
