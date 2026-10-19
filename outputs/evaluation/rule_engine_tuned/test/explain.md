<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_engine_tuned/test - test một lần cho rule engine đã tune

> **Dùng để làm gì:** Test split của rule engine dùng cấu hình tuned (`scratch 27`: accentless 0.0, obfuscated_combined 0.45, negation 1.0), chấm đúng một lần với threshold đã đóng băng từ validation (plan §44). Lần đo này thuộc **phiên bản lexicon 668 entry** (bản cũ 1828 entry từng cho macro-F1 0.4927).

> **Sinh ra bởi:** `python scripts/17_evaluate_rule_engine.py --config configs/rule_engine_tuned.yaml --split test --out outputs/evaluation/rule_engine_tuned`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `metrics.json` | Chỉ số test với cấu hình tuned và threshold đóng băng (macro-F1 0.5335). |
| `*.csv / *.txt / *.json` | Confusion matrix, per-class, FP/FN, latency, robustness như baseline. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `classification_report.txt` | file | 389 B · 2026-10-19 13:11:41 |
| `confusion_matrix.csv` | file | 119 B · 2026-10-19 13:11:41 |
| `confusion_matrix.png` | file | 50.3 KB · 2026-10-19 13:12:13 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-19 13:11:41 |
| `confusion_matrix_normalized.png` | file | 58.1 KB · 2026-10-19 13:12:13 |
| `explain.md` | file | 3.2 KB · 2026-10-19 15:06:21 |
| `false_negative_analysis.csv` | file | 57.1 KB · 2026-10-19 13:11:41 |
| `false_positive_analysis.csv` | file | 60.6 KB · 2026-10-19 13:11:41 |
| `latency.json` | file | 137 B · 2026-10-19 13:11:41 |
| `latency_distribution.png` | file | 69.4 KB · 2026-10-19 13:12:16 |
| `metrics.json` | file | 1.9 KB · 2026-10-19 13:11:41 |
| `per_class_f1.png` | file | 42.2 KB · 2026-10-19 13:12:13 |
| `per_class_metrics.csv` | file | 288 B · 2026-10-19 13:11:41 |
| `pr_curve.png` | file | 116.8 KB · 2026-10-19 13:12:14 |
| `predictions.csv` | file | 564.7 KB · 2026-10-19 13:11:41 |
| `robustness.json` | file | 805 B · 2026-10-19 13:12:16 |
| `robustness_detection.png` | file | 70.3 KB · 2026-10-19 13:12:16 |
| `roc_curve.png` | file | 145.7 KB · 2026-10-19 13:12:14 |
| `rule_scores.csv` | file | 174.4 KB · 2026-10-19 13:12:11 |
| `threshold_sweep.png` | file | 90.5 KB · 2026-10-19 13:12:14 |

## Cách đọc

1. So sánh với `evaluation/rule_engine_tuned/metrics.json` (validation 0.5481) để thấy gap validation -> test.
2. Đánh đổi robustness (`robustness.json`): no_accent giảm vì accentless_alias = 0; leet/spaced/repeat tăng nhờ threshold HATE thấp hơn (0.64).

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
