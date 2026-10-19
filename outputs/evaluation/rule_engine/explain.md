<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_engine - đánh giá rule engine

> **Dùng để làm gì:** Kết quả chấm riêng cho rule engine từ vựng (không dùng model AI). Threshold quyết định được tune **trên validation** rồi đóng băng vào `thresholds.json`; test chỉ được chấm một lần với threshold đã đóng băng (plan §44).

> **Sinh ra bởi:** `python scripts/17_evaluate_rule_engine.py` và `python scripts/19_plot_rule_engine.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `thresholds.json` | Ngưỡng riêng từng nhãn tối ưu trên validation + vết tìm kiếm; đóng băng trước khi chấm test. |
| `metrics.json` | Bộ chỉ số đầy đủ: accuracy, macro/weighted F1, per-class P/R/F1, ROC-AUC/AP, confusion matrix. |
| `per_class_metrics.csv` | Precision/recall/F1/support/AUC/AP theo từng nhãn và dòng MACRO. |
| `confusion_matrix.csv` | Ma trận nhầm lẫn dạng CSV (hàng = nhãn thật). |
| `confusion_matrix.txt` | Ma trận nhầm lẫn và bản chuẩn hoá theo hàng, in tên nhãn. |
| `classification_report.txt` | Báo cáo phân loại đầy đủ của sklearn (tên nhãn, không dùng id). |
| `false_positive_analysis.csv` | Các câu CLEAN bị báo nhầm kèm bằng chứng khớp mạnh nhất. |
| `false_negative_analysis.csv` | Các câu HATE/OFFENSIVE bị bỏ sót. |
| `latency.json` | Latency (mean/p50/p95/max) và throughput mỗi câu. |
| `robustness.json` | Tỉ lệ phát hiện trên mẫu bị làm nhiễu (clean/leet/spaced/no_accent/repeat) kèm `detection_drop_vs_clean` (do script 19 bổ sung dòng baseline `clean`). |
| `rule_scores.csv` | Điểm `rule_hate_score`/`rule_offensive_score` và nhãn dự đoán per-sample; đầu vào cho bước fusion (script 19 sinh). |
| `confusion_matrix.png` | Hình ma trận nhầm lẫn thô (script 19). |
| `confusion_matrix_normalized.png` | Hình ma trận nhầm lẫn chuẩn hoá theo hàng (script 19). |
| `per_class_f1.png` | Cột F1 theo từng lớp (script 19). |
| `pr_curve.png` | Đường Precision-Recall one-vs-rest + macro (script 19). |
| `roc_curve.png` | Đường ROC one-vs-rest + macro (script 19). |
| `threshold_sweep.png` | macro-F1 theo từng bước tune threshold trên validation (script 19). |
| `robustness_detection.png` | Cột detection rate theo biến đổi, gồm baseline `clean` (script 19). |
| `latency_distribution.png` | Histogram latency mỗi câu + p50/p95/mean (script 19). |
| `test/` | Kết quả chấm **test một lần** với threshold đã đóng băng, gồm cả PNG và `rule_scores.csv` (nằm riêng để không ghi đè kết quả validation). |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `test` | thư mục | 20 file, 1.5 MB |
| `test_legacy922` | thư mục | 9 file, 154.6 KB |
| `classification_report.txt` | file | 389 B · 2026-10-19 13:07:20 |
| `confusion_matrix.csv` | file | 117 B · 2026-10-19 13:07:20 |
| `confusion_matrix.png` | file | 54.9 KB · 2026-10-19 13:49:53 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-19 13:07:20 |
| `confusion_matrix_normalized.png` | file | 59.1 KB · 2026-10-19 13:49:53 |
| `explain.md` | file | 5.6 KB · 2026-10-19 15:06:21 |
| `false_negative_analysis.csv` | file | 38.8 KB · 2026-10-19 13:07:20 |
| `false_positive_analysis.csv` | file | 71.6 KB · 2026-10-19 13:07:20 |
| `latency.json` | file | 136 B · 2026-10-19 13:07:20 |
| `latency_distribution.png` | file | 72.2 KB · 2026-10-19 13:49:56 |
| `metrics.json` | file | 1.9 KB · 2026-10-19 13:07:20 |
| `per_class_f1.png` | file | 41.7 KB · 2026-10-19 13:49:54 |
| `per_class_metrics.csv` | file | 284 B · 2026-10-19 13:07:20 |
| `pr_curve.png` | file | 109.9 KB · 2026-10-19 13:49:54 |
| `predictions.csv` | file | 240.8 KB · 2026-10-19 13:07:20 |
| `robustness.json` | file | 798 B · 2026-10-19 13:49:56 |
| `robustness_detection.png` | file | 67.2 KB · 2026-10-19 13:49:56 |
| `roc_curve.png` | file | 144.7 KB · 2026-10-19 13:49:54 |
| `rule_scores.csv` | file | 72.8 KB · 2026-10-19 13:49:51 |
| `threshold_sweep.png` | file | 88.8 KB · 2026-10-19 13:49:55 |
| `thresholds.json` | file | 2.0 KB · 2026-10-19 13:07:20 |

## Cách đọc

1. Số liệu chính (lexicon 668 entry): validation macro-F1 **0.4555** (HATE 0.3677, OFFENSIVE 0.1686, acc 0.7021), thresholds [0.85, 0.0, 0.41].
2. `test/metrics.json` = **0.4536** (acc 0.7070, HATE 0.3717, OFFENSIVE 0.1546) - đây là baseline để so với bản tuned 0.5335 (**+0.0799 macro-F1**).
3. Số liệu chính: `metrics.json` (macro F1) và `per_class_metrics.csv`.
4. Xem vì sao báo nhầm: `false_positive_analysis.csv` (cột `top_matches`).
5. So sánh AUC/AP giữa các lớp: cột `roc_auc`, `average_precision` trong `per_class_metrics.csv`.
6. Xem hình: `confusion_matrix*.png`, `per_class_f1.png`, `pr_curve.png`, `roc_curve.png`, `robustness_detection.png`, `latency_distribution.png`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
