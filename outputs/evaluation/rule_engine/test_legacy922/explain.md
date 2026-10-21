<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_engine/test_legacy922 - số test của phiên bản lexicon 922 entry (lưu trữ)

> **Dùng để làm gì:** Kết quả chấm test **bản cũ** của rule engine baseline khi `verified_lexicon.csv` còn 922 entry (macro-F1 0.4667). Khi hệ thống được dựng lại với lexicon 668 entry, thư mục `test/` phải chấm lại một lần nên bản cũ được chuyển vào đây để không mất lịch sử (plan §44).

> **Sinh ra bởi:** Kết quả của lần chạy cũ, chỉ được **di chuyển** (không sinh lại được).

> **Quy tắc giữ lại:** **KHÔNG XÓA, KHÔNG CHẠY LẠI** - đây là lịch sử đo lường (mỗi lần đo test chỉ hợp lệ một lần).

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `metrics.json` | Chỉ số test bản lexicon 922: macro-F1 0.4667, acc 0.7201. |
| `thresholds.json` | Ngưỡng đóng băng thời điểm đó (khác với ngưỡng hiện tại [0.85, 0.0, 0.41]). |
| `*.csv / *.txt` | Confusion matrix, per-class, FP/FN, latency bản cũ. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `classification_report.txt` | file | 389 B · 2026-10-19 09:47:22 |
| `confusion_matrix.csv` | file | 120 B · 2026-10-19 09:47:22 |
| `confusion_matrix.txt` | file | 452 B · 2026-10-19 09:47:22 |
| `explain.md` | file | 2.3 KB · 2026-10-20 07:13:20 |
| `false_negative_analysis.csv` | file | 56.7 KB · 2026-10-19 09:47:22 |
| `false_positive_analysis.csv` | file | 92.4 KB · 2026-10-19 09:47:22 |
| `latency.json` | file | 137 B · 2026-10-19 09:47:22 |
| `metrics.json` | file | 1.9 KB · 2026-10-19 09:47:22 |
| `per_class_metrics.csv` | file | 286 B · 2026-10-19 09:47:22 |

## Cách đọc

1. **Chỉ dùng để đối chiếu lịch sử**, không dùng làm baseline cho kết quả hiện tại - baseline hiện tại là `evaluation/rule_engine/test/` (lexicon 668, macro-F1 0.4536).

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:45. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
