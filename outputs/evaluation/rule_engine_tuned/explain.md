<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_engine_tuned - rule engine đã tune trọng số

> **Dùng để làm gì:** Cùng bộ artifact như `evaluation/rule_engine/` nhưng dùng **`configs/rule_engine_tuned.yaml`**: các trọng số `scoring.*` được tune bằng coordinate descent trên **validation** (`scripts/27_tune_rule_weights.py`, hồ sơ từng trial ở `tuning_trials.csv` của chính folder này). Ba tham số đổi so với bản gốc: `match_factors.accentless_alias` 0.90 -> 0.00 (view không dấu là tín hiệu phụ nhưng lại khớp nhầm âm tiết phổ biến, plan §38), `match_factors.obfuscated_combined` 0.60 -> 0.45, `negation_factor` 0.50 -> 1.00. Threshold tune trên validation rồi đóng băng cho test (plan §44). Đây là cấu hình rule engine tốt nhất, không ghi đè baseline.

> **Sinh ra bởi:** `python scripts/17_evaluate_rule_engine.py --config configs/rule_engine_tuned.yaml --out outputs/evaluation/rule_engine_tuned` (và `--split test` cho bản test) + `python scripts/19_plot_rule_engine.py --config configs/rule_engine_tuned.yaml --out outputs/evaluation/rule_engine_tuned`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `thresholds.json` | Ngưỡng tune trên validation với trọng số đã tune; đóng băng trước khi chấm test. |
| `metrics.json` | Chỉ số validation của cấu hình tuned (macro-F1 0.5481). |
| `rule_scores.csv` | Điểm rule per-sample (script 19) - đầu vào của bước fusion. |
| `tuning_trials.csv` | 155 trial của coordinate descent (script 27) - bằng chứng trọng số được tune trên **validation** (plan §44). Đây là bản sao lưu từ nhật ký tune gốc; sau khi dọn cây (chỉ giữ baseline + tuned) đây là nơi duy nhất còn `trials.csv`. |
| `tuning_report.json` | Tóm tắt phiên tune: baseline 0.4555 -> 0.5481, các trọng số đã đổi (`changed`), SHA-256 từ điển. |
| `test/` | Chấm **test một lần** với threshold đóng băng (macro-F1 0.5335), gồm metrics, confusion, per-class, FP/FN, latency, robustness. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `test` | thư mục | 20 file, 1.5 MB |
| `classification_report.txt` | file | 389 B · 2026-10-19 13:07:20 |
| `confusion_matrix.csv` | file | 116 B · 2026-10-19 13:07:20 |
| `confusion_matrix.png` | file | 54.7 KB · 2026-10-19 13:50:05 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-19 13:07:20 |
| `confusion_matrix_normalized.png` | file | 58.9 KB · 2026-10-19 13:50:05 |
| `explain.md` | file | 5.2 KB · 2026-10-19 15:06:21 |
| `false_negative_analysis.csv` | file | 41.1 KB · 2026-10-19 13:07:20 |
| `false_positive_analysis.csv` | file | 27.9 KB · 2026-10-19 13:07:20 |
| `latency.json` | file | 137 B · 2026-10-19 13:07:20 |
| `latency_distribution.png` | file | 72.0 KB · 2026-10-19 13:50:08 |
| `metrics.json` | file | 1.9 KB · 2026-10-19 13:07:20 |
| `per_class_f1.png` | file | 42.8 KB · 2026-10-19 13:50:05 |
| `per_class_metrics.csv` | file | 289 B · 2026-10-19 13:07:20 |
| `pr_curve.png` | file | 118.4 KB · 2026-10-19 13:50:06 |
| `predictions.csv` | file | 225.3 KB · 2026-10-19 13:07:20 |
| `robustness.json` | file | 802 B · 2026-10-19 13:50:07 |
| `robustness_detection.png` | file | 67.5 KB · 2026-10-19 13:50:07 |
| `roc_curve.png` | file | 148.0 KB · 2026-10-19 13:50:06 |
| `rule_scores.csv` | file | 69.2 KB · 2026-10-19 13:50:03 |
| `threshold_sweep.png` | file | 90.5 KB · 2026-10-19 13:50:06 |
| `thresholds.json` | file | 1.9 KB · 2026-10-19 13:07:20 |
| `tuning_report.json` | file | 2.4 KB · 2026-10-19 13:06:04 |
| `tuning_trials.csv` | file | 15.5 KB · 2026-10-19 13:06:04 |

## Cách đọc

1. So `metrics.json` (validation) với `evaluation/rule_engine/metrics.json` (baseline accentless=0.90): macro-F1 0.5481 vs 0.4555, HATE 0.4967 vs 0.3677, OFFENSIVE 0.2310 vs 0.1686.
2. `test/metrics.json` là con số test cuối: macro-F1 **0.5335**, accuracy 0.8398, HATE 0.4323, OFFENSIVE 0.2418 (val 0.5481 -> test 0.5335, gap lành mạnh) - so baseline cùng lexicon 668 `evaluation/rule_engine/test/metrics.json` = 0.4536 (**+0.0799**).
3. Đánh đổi phải nhớ (`robustness.json`, 400 mẫu nhiễu): `no_accent` giảm - validation 0.3625 (baseline 0.530), test 0.325 (baseline 0.540) - vì `accentless_alias = 0`; bù lại leet/spaced/repeat **tốt hơn** baseline (test: 0.460/0.605/0.5425 vs 0.440/0.4725/0.3775).
4. Bộ artifact chi tiết giống hệt `evaluation/rule_engine/` (confusion matrix, per-class, PR/ROC, robustness...).

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
