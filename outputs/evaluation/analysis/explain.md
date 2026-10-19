<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/analysis - báo cáo phân tích hậu nghiệm

> **Dùng để làm gì:** Ba báo cáo phân tích đọc từ **test predictions đã đóng băng** của 4 model (không huấn luyện, không tune gì thêm): (1) chống trùng lặp giữa các split - `leakage_report`; (2) kiểm định thống kê post-hoc - CI bootstrap 95% từng model, so cặp trên test - `significance_report`; (3) supplementary (calibration ECE/Brier/NLL, MCC...) - `supplementary_report`, được `papers/supplementary/main.tex` trỏ tới.

> **Sinh ra bởi:** `python scripts/08_check_leakage.py` (leakage), `python scripts/06_analyze_significance.py` (significance), `python scripts/11_supplementary_analyses.py` (supplementary)

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `leakage_report.md / .json` | Trùng lặp giữa train/validation/test (895/6.680 comment test trùng train/val = 13.4%) + tập de-leaked 5.785 mẫu. |
| `significance_report.md` | CI bootstrap accuracy/macro-F1 từng model + so cặp - nguồn cho mọi tuyên bố 'model nào hơn' trên test. |
| `significance_summary.json` | Cùng nội dung ở dạng máy-đọc (CI, pairwise delta, p-value). |
| `supplementary_report.md` | Calibration (ECE/Brier/NLL), MCC... dạng Markdown cho bài báo. |
| `supplementary_summary.json` | Dữ liệu của supplementary ở dạng máy-đọc. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `explain.md` | file | 3.0 KB · 2026-10-19 15:06:21 |
| `leakage_report.json` | file | 1.8 KB · 2026-10-19 14:15:48 |
| `leakage_report.md` | file | 805 B · 2026-10-19 14:15:48 |
| `significance_report.md` | file | 2.0 KB · 2026-10-19 14:15:48 |
| `significance_summary.json` | file | 46.3 KB · 2026-10-19 14:15:48 |
| `supplementary_report.md` | file | 1.2 KB · 2026-10-19 14:15:48 |
| `supplementary_summary.json` | file | 20.2 KB · 2026-10-19 14:15:48 |

## Cách đọc

1. Nghi ngờ leakage: đọc `leakage_report.md` trước - số trùng test/train quyết định mức độ tin số test.
2. Muốn tuyên bố model A hơn model B: mở `significance_report.md`, xem CI của Δ và p - không đọc p từ bảng metric.
3. Bài supplementary trỏ thẳng vào `supplementary_report.md` - đừng đổi tên file này.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
