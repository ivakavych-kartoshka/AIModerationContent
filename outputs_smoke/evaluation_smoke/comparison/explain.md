<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/comparison - bảng và biểu đồ so sánh

> **Dùng để làm gì:** Nơi dễ dễ nhất để đọc kết quả so sánh: một bảng tất cả chỉ số của 4 model (CSV và Markdown), cùng các biểu đồ PR/ROC/validation dùng cho bài báo. Hai thư mục con: `rule_engine/` (so sánh baseline vs tuned) và `vi_vs_stage2/` (so 2 run cùng model).

> **Sinh ra bởi:** `python scripts/03_compare_models.py` (4 model); bộ so sánh 2 rule engine do scripts 26 + 28 ghi vào thư mục con `rule_engine/`

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
| `accuracy_all_models.png` | file | 67.8 KB · 2026-10-05 07:20:32 |
| `comparison_summary.json` | file | 4.9 KB · 2026-10-05 07:20:32 |
| `explain.md` | file | 4.5 KB · 2026-10-20 07:13:19 |
| `macro_f1_all_models.png` | file | 72.8 KB · 2026-10-05 07:20:32 |
| `model_comparison.csv` | file | 2.3 KB · 2026-10-05 07:20:30 |
| `model_comparison.md` | file | 4.4 KB · 2026-10-05 07:20:30 |
| `per_class_f1_all_models.png` | file | 58.7 KB · 2026-10-05 07:20:32 |
| `pr_clean_all_models.png` | file | 78.7 KB · 2026-10-05 07:20:31 |
| `pr_curve_all_models.png` | file | 124.9 KB · 2026-10-05 07:20:31 |
| `pr_hate_all_models.png` | file | 133.1 KB · 2026-10-05 07:20:31 |
| `pr_offensive_all_models.png` | file | 128.7 KB · 2026-10-05 07:20:31 |
| `roc_clean_all_models.png` | file | 109.1 KB · 2026-10-05 07:20:31 |
| `roc_curve_all_models.png` | file | 126.5 KB · 2026-10-05 07:20:31 |
| `roc_hate_all_models.png` | file | 112.3 KB · 2026-10-05 07:20:31 |
| `roc_offensive_all_models.png` | file | 117.4 KB · 2026-10-05 07:20:31 |
| `validation_accuracy_all_models.png` | file | 91.5 KB · 2026-10-05 07:20:31 |
| `validation_loss_all_models.png` | file | 96.3 KB · 2026-10-05 07:20:32 |

## Cách đọc

1. Muốn so 2 rule engine (baseline vs tuned): đọc `rule_engine/rule_engine_baseline_vs_tuned.md` - đủ bảng số, hình kèm và kết luận kiểm định.
2. Bảng quan trọng nhất của 4 model: `model_comparison.md` (macro F1, weighted F1, per-class F1, latency).
3. Giải thích vì sao accuracy không phải số liệu chính: xem mục 2.2 của `README.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
