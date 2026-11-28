<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_engine/test - chấm test một lần cho rule engine

> **Dùng để làm gì:** Cùng bộ artifact như `evaluation/rule_engine/` nhưng cho **test split**, chấm đúng một lần với threshold đã đóng băng từ validation (plan §44). Nằm riêng để không bao giờ ghi đè kết quả validation đã dùng để tune.

> **Sinh ra bởi:** `python scripts/17_evaluate_rule_engine.py --split test` và `python scripts/19_plot_rule_engine.py --split test`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `metrics.json` | Chỉ số test với threshold đóng băng (dùng cho ablation §43). |
| `per_class_metrics.csv` | Precision/recall/F1/support/AUC/AP theo lớp trên test. |
| `confusion_matrix.csv / .txt` | Ma trận nhầm lẫn thô và chuẩn hoá theo hàng. |
| `classification_report.txt` | Báo cáo phân loại sklearn trên test. |
| `false_positive_analysis.csv / false_negative_analysis.csv` | Phân tích lỗi kèm bằng chứng khớp mạnh nhất. |
| `latency.json` | Latency/throughput trên test. |
| `robustness.json` | Detection rate theo biến đổi + baseline `clean` và `detection_drop_vs_clean`. |
| `rule_scores.csv` | Điểm rule per-sample trên test; đầu vào cho bước fusion. |
| `*.png` | Confusion matrix, per-class F1, PR/ROC, threshold sweep, robustness, latency. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `classification_report.txt` | file | 389 B · 2026-10-19 13:38:30 |
| `confusion_matrix.csv` | file | 120 B · 2026-10-19 13:38:30 |
| `confusion_matrix.png` | file | 57.1 KB · 2026-10-19 13:38:37 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-19 13:38:30 |
| `confusion_matrix_normalized.png` | file | 59.6 KB · 2026-10-19 13:38:37 |
| `explain.md` | file | 4.1 KB · 2026-10-20 07:13:20 |
| `false_negative_analysis.csv` | file | 55.9 KB · 2026-10-19 13:38:30 |
| `false_positive_analysis.csv` | file | 93.9 KB · 2026-10-19 13:38:30 |
| `latency.json` | file | 135 B · 2026-10-19 13:38:30 |
| `latency_distribution.png` | file | 76.5 KB · 2026-10-19 13:38:39 |
| `metrics.json` | file | 1.8 KB · 2026-10-19 13:38:30 |
| `per_class_f1.png` | file | 40.7 KB · 2026-10-19 13:38:38 |
| `per_class_metrics.csv` | file | 289 B · 2026-10-19 13:38:30 |
| `pr_curve.png` | file | 107.5 KB · 2026-10-19 13:38:38 |
| `predictions.csv` | file | 603.5 KB · 2026-10-19 13:38:30 |
| `robustness.json` | file | 801 B · 2026-10-19 13:38:39 |
| `robustness_detection.png` | file | 66.6 KB · 2026-10-19 13:38:39 |
| `roc_curve.png` | file | 144.5 KB · 2026-10-19 13:38:38 |
| `rule_scores.csv` | file | 183.5 KB · 2026-10-19 13:38:36 |
| `threshold_sweep.png` | file | 88.8 KB · 2026-10-19 13:38:38 |

## Cách đọc

1. Đây là con số **test cuối cùng** của rule engine baseline: macro-F1 **0.4536**, acc 0.7070, HATE 0.3717, OFFENSIVE 0.1546 (lexicon 668 entry, thresholds đóng băng [0.85, 0.0, 0.41]); không tune lại sau khi có kết quả này.
2. So với `evaluation/rule_engine_tuned/test/metrics.json` (0.5335): chênh lệch **+0.0799 macro-F1** đến từ tune trọng số, không phải từ điển hay ngưỡng khác.
3. Bản test cũ chạy trên lexicon 922 entry (macro-F1 0.4667) được lưu ở `test_legacy922/` để không ghi đè lịch sử.
4. Robustness test (400 mẫu nhiễu): clean 0.555, leet 0.440, spaced 0.4725, no_accent 0.540, repeat 0.3775.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:45. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
