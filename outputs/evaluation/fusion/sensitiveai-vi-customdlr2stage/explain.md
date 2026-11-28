<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/fusion/<model> - fusion cho một model AI

> **Dùng để làm gì:** Một thư mục cho mỗi model AI: alpha/threshold đã đóng băng trên validation + metric test của 6 hệ thống (AI-only argmax, rule-only, fusion argmax/thresholds/per-channel/meta).

> **Sinh ra bởi:** `python scripts/23_evaluate_fusion.py --model <model>`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `fusion_report.md` | Bảng so sánh test của các hệ thống + tham số đã đóng băng. |
| `metrics.json` | Chỉ số đầy đủ từng biến thể (validation + test). |
| `fusion_predictions.csv` | Dự đoán per-sample của biến thể `primary`. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `explain.md` | file | 2.0 KB · 2026-10-20 07:13:20 |
| `fusion_predictions.csv` | file | 1.2 MB · 2026-10-19 13:14:27 |
| `fusion_report.md` | file | 1.3 KB · 2026-10-19 13:14:27 |
| `metrics.json` | file | 26.5 KB · 2026-10-19 13:14:27 |

## Cách đọc

1. Cột `macro-F1` trong `fusion_report.md`: so `ai_only_argmax` với biến thể `primary`.
2. `primary` chọn theo macro-F1 **validation**, nên đôi khi kém hơn một biến thể khác trên test - hệ quả của việc chọn tham số trên validation, không phải lỗi.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:45. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
