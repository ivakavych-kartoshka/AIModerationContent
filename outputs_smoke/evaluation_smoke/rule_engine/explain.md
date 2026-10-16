<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_engine - đánh giá rule engine

> **Dùng để làm gì:** Kết quả chấm riêng cho rule engine từ vựng (không dùng model AI). Threshold quyết định được tune **trên validation** rồi đóng băng vào `thresholds.json`; test chỉ được chấm một lần với threshold đã đóng băng (plan §44).

> **Sinh ra bởi:** `python scripts/17_evaluate_rule_engine.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

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
| `robustness.json` | Tỉ lệ phát hiện trên mẫu bị làm nhiễu (leet/spaced/no_accent/repeat) nếu chạy `--robustness`. |
| `test/` | Kết quả chấm **test một lần** với threshold đã đóng băng (nằm riêng để không ghi đè kết quả validation). |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `smoke_test.json` | file | 4.5 KB · 2026-10-17 00:20:05 |
| `smoke_test.md` | file | 2.5 KB · 2026-10-17 00:20:05 |

## Cách đọc

1. Số liệu chính: `metrics.json` (macro F1) và `per_class_metrics.csv`.
2. Xem vì sao báo nhầm: `false_positive_analysis.csv` (cột `top_matches`).
3. So sánh AUC/AP giữa các lớp: cột `roc_auc`, `average_precision` trong `per_class_metrics.csv`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
