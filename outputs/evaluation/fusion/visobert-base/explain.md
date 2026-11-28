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
| `rule_silence.json` | Ablation quy ước `rule im lặng` (chỉ model chính): so cấu hình đã ship với biến thể `neutral` (rule im lặng thì cho điểm AI đi thẳng), kèm paired bootstrap + McNemar và bảng đối chiếu correction/regression. |
| `explain.md` | File này. |

## Cách đọc

1. Cột `macro-F1` trong `fusion_report.md`: so `ai_only_argmax` với biến thể `primary`.
2. `primary` chọn theo macro-F1 **validation**, nên đôi khi kém hơn một biến thể khác trên test - hệ quả của việc chọn tham số trên validation, không phải lỗi.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-11-20 05:02:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
