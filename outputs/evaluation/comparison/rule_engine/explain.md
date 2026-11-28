<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/comparison/rule_engine - so sánh 2 rule engine

> **Dùng để làm gì:** Bộ so sánh baseline (`rule_engine`) với bản thắng tune (`rule_engine_tuned`): báo Markdown 8 mục (khác biệt cấu hình + ngưỡng đóng băng, metric validation/test, chi tiết per-class, robustness theo mode nhiễu, độ trễ, kiểm định paired bootstrap + McNemar, kết luận) kèm 3 hình và số kiểm định. Tất cả đều đọc artifact đã đóng băng - không chấm lại split nào (plan §44).

> **Sinh ra bởi:** `python scripts/26_rule_significance.py --a outputs/evaluation/rule_engine_tuned/test/predictions.csv --b outputs/evaluation/rule_engine/test/predictions.csv --out .../rule_engine_significance.json` rồi `python scripts/28_compare_rule_engines.py` (plan §43-44)

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `rule_engine_baseline_vs_tuned.md` | Báo chính 8 mục - từ khác biệt cấu hình đến kết luận có kiểm định; nhúng 3 hình bên dưới. |
| `rule_engine_significance.json` | Số kiểm định Δ trên test (CI 95%, p, McNemar) do `scripts/26` sinh - nguồn cho mục 7 của báo cáo. |
| `rule_engine_macro_f1.png` | Thanh so macro-F1 và accuracy của 2 hệ thống, validation và test. |
| `rule_engine_per_class_f1.png` | F1 từng lớp (HATE/OFFENSIVE/CLEAN) của 2 hệ thống, validation và test. |
| `rule_engine_robustness.png` | Tỉ lệ phát hiện theo 5 chế độ nhiễu của 2 hệ thống, validation và test. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `explain.md` | file | 3.0 KB · 2026-10-20 07:13:20 |
| `rule_engine_baseline_vs_tuned.md` | file | 5.0 KB · 2026-10-19 15:09:36 |
| `rule_engine_macro_f1.png` | file | 45.5 KB · 2026-10-19 15:09:35 |
| `rule_engine_per_class_f1.png` | file | 51.9 KB · 2026-10-19 15:09:35 |
| `rule_engine_robustness.png` | file | 70.6 KB · 2026-10-19 15:09:36 |
| `rule_engine_significance.json` | file | 3.4 KB · 2026-10-19 14:22:18 |

## Cách đọc

1. Đọc `rule_engine_baseline_vs_tuned.md` từ đầu đến cuối là đủ cho báo cáo; hình đã nhúng theo từng mục.
2. Mục 7-8 (kiểm định + kết luận) là phần cần trích khi mentor hỏi 'cải thiện có thật không'.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:45. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
