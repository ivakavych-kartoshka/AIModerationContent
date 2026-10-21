<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/comparison - bảng và biểu đồ so sánh

> **Dùng để làm gì:** Nơi dễ dễ nhất để đọc kết quả so sánh: một bảng tất cả chỉ số của 4 model (CSV và Markdown), cùng các biểu đồ PR/ROC/validation dùng cho bài báo. Hai thư mục con: `rule_engine/` (so sánh baseline vs tuned) và `vi_vs_stage2/` (so 2 run cùng model).

> **Sinh ra bởi:** `python scripts/03_compare_models.py` (4 model); bộ so sánh 2 rule engine do scripts 26 + 28 ghi vào thư mục con `rule_engine/`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

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
| `rule_engine` | thư mục | 6 file, 179.3 KB |
| `vi_vs_stage2` | thư mục | 17 file, 1.1 MB |
| `accuracy_all_models.png` | file | 61.6 KB · 2026-10-14 23:56:09 |
| `comparison_summary.json` | file | 7.5 KB · 2026-10-14 23:56:09 |
| `explain.md` | file | 4.5 KB · 2026-10-20 07:13:20 |
| `f1_summary_all_models.png` | file | 67.1 KB · 2026-10-15 01:27:23 |
| `macro_f1_all_models.png` | file | 62.8 KB · 2026-10-14 23:56:09 |
| `model_comparison.csv` | file | 3.4 KB · 2026-10-14 23:56:05 |
| `model_comparison.md` | file | 5.9 KB · 2026-10-14 23:56:05 |
| `per_class_f1_all_models.png` | file | 54.9 KB · 2026-10-14 23:56:09 |
| `pr_clean_all_models.png` | file | 103.1 KB · 2026-10-14 23:56:07 |
| `pr_curve_all_models.png` | file | 151.4 KB · 2026-10-14 23:56:05 |
| `pr_hate_all_models.png` | file | 158.1 KB · 2026-10-14 23:56:06 |
| `pr_offensive_all_models.png` | file | 160.7 KB · 2026-10-14 23:56:06 |
| `roc_clean_all_models.png` | file | 137.1 KB · 2026-10-14 23:56:07 |
| `roc_curve_all_models.png` | file | 160.3 KB · 2026-10-14 23:56:05 |
| `roc_hate_all_models.png` | file | 127.8 KB · 2026-10-14 23:56:06 |
| `roc_offensive_all_models.png` | file | 136.0 KB · 2026-10-14 23:56:07 |
| `validation_accuracy_all_models.png` | file | 210.3 KB · 2026-10-14 23:56:08 |
| `validation_loss_accuracy_all_models.png` | file | 156.5 KB · 2026-10-14 23:56:08 |
| `validation_loss_all_models.png` | file | 137.2 KB · 2026-10-14 23:56:08 |

## Cách đọc

1. Muốn so 2 rule engine (baseline vs tuned): đọc `rule_engine/rule_engine_baseline_vs_tuned.md` - đủ bảng số, hình kèm và kết luận kiểm định.
2. Bảng quan trọng nhất của 4 model: `model_comparison.md` (macro F1, weighted F1, per-class F1, latency).
3. Giải thích vì sao accuracy không phải số liệu chính: xem mục 2.2 của `README.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:45. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
