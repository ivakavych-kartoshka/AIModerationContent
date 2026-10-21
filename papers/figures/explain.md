<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# papers/figures - hình chính của bài báo

> **Dùng để làm gì:** Hình PNG của bài báo: phần lớn **copy tự động** từ `outputs/evaluation/comparison/` (script 03) để `main.tex` chèn được; riêng `system_architecture.png` do script 30 vẽ trực tiếp. Không vẽ tay.

> **Sinh ra bởi:** `python scripts/03_compare_models.py` (hình so sánh model); `python scripts/30_plot_system_architecture.py` (sơ đồ kiến trúc toàn hệ thống)

> **Quy tắc giữ lại:** Sinh lại mỗi lần chạy `03_compare_models.py`; sửa tay sẽ bị ghi đè.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `macro_f1_all_models.png / accuracy_all_models.png` | So sánh tổng hợp giữa các model. |
| `per_class_f1_all_models.png / f1_summary_all_models.png` | F1 theo từng lớp. |
| `confusion_matrices.png` | Lưới confusion matrix của các model. |
| `pr_*.png / roc_*.png` | Đường PR/ROC theo từng nhãn và macro. |
| `validation_loss_all_models.png / validation_accuracy_all_models.png` | Learning curve trên validation. |
| `system_architecture.png` | Sơ đồ kiến trúc toàn hệ thống: 2 nhánh (AI mDeBERTa-v3 + multi-view head + rule engine lexicon 668 term) hợp nhất qua fusion p'=p+βr (AI trọng số 1, rule hạ trọng số β) rồi ngưỡng đóng băng - dành cho mục Methodology của bài báo. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `confusion_matrices.png` | file | 76.4 KB · 2026-10-15 22:02:42 |
| `explain.md` | file | 3.3 KB · 2026-10-20 07:13:19 |
| `f1_summary_all_models.png` | file | 67.1 KB · 2026-10-15 01:27:23 |
| `kien_truc_hẹ_thong.png` | file | 1.4 MB · 2026-10-19 15:46:00 |
| `macro_f1_all_models.png` | file | 62.8 KB · 2026-10-14 23:56:09 |
| `per_class_f1_all_models.png` | file | 54.9 KB · 2026-10-14 23:56:09 |
| `pr_clean_all_models.png` | file | 103.1 KB · 2026-10-14 23:56:09 |
| `pr_curve_all_models.png` | file | 151.4 KB · 2026-10-14 23:56:09 |
| `pr_hate_all_models.png` | file | 158.1 KB · 2026-10-14 23:56:09 |
| `pr_offensive_all_models.png` | file | 160.7 KB · 2026-10-14 23:56:09 |
| `robustness.png` | file | 36.4 KB · 2026-10-19 08:30:43 |
| `roc_clean_all_models.png` | file | 137.1 KB · 2026-10-14 23:56:09 |
| `roc_curve_all_models.png` | file | 160.3 KB · 2026-10-14 23:56:09 |
| `roc_hate_all_models.png` | file | 127.8 KB · 2026-10-14 23:56:09 |
| `roc_offensive_all_models.png` | file | 136.0 KB · 2026-10-14 23:56:09 |
| `system_architecture.png` | file | 250.8 KB · 2026-10-20 06:39:47 |
| `validation_accuracy_all_models.png` | file | 210.3 KB · 2026-10-14 23:56:09 |
| `validation_loss_accuracy_all_models.png` | file | 156.5 KB · 2026-10-14 23:56:09 |
| `validation_loss_all_models.png` | file | 137.2 KB · 2026-10-14 23:56:09 |

## Cách đọc

1. File ở đây là **bản copy**; số liệu gốc nằm ở `outputs/evaluation/comparison/` (riêng `system_architecture.png` do script 30 sinh trực tiếp).

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
