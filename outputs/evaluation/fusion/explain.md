<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/fusion - hybrid fusion (rule engine + AI)

> **Dùng để làm gì:** Kết quả ghép điểm rule engine với xác suất AI (plan §40-41): `alpha` (trọng số AI), threshold và meta-classifier đều tune trên **validation**, đóng băng rồi chấm test **một lần** (plan §44). Mỗi model AI một thư mục con, `comparison.md` là bảng tổng hợp. Kết luận hiện tại (lexicon 668 entry, rule engine test macro-F1 0.5335 so với AI-only 0.65+): chênh lệch fusion - AI nằm trong dải nhiễu với 5/6 model (Δ macro-F1 từ -0.0085 đến +0.0034); riêng `sensitiveai-vi-custom-curriculum` fusion **tốt hơn có ý nghĩa** (+0.0057, p=0.033, McNemar p=0.0005) còn `sensitiveai-vi-stage2` kém hơn (-0.0085) nhưng không có ý nghĩa. Chưa đủ bằng chứng khẳng định fusion cải thiện nhất quán - xem `significance_report.md`.

> **Sinh ra bởi:** `python scripts/23_evaluate_fusion.py` rồi `python scripts/25_fusion_significance.py` (plan §43-44); hình so sánh: `python scripts/29_plot_fusion_comparison.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `comparison.md` | Bảng tổng hợp: alpha tốt nhất trên validation, macro-F1 test của AI-only vs các biến thể fusion, biến thể `primary` và Δ so với AI-only cho từng model. |
| `fusion_ai_vs_fusion.png` | Hình 1: thanh so macro-F1 test của AI-only vs fusion primary từng model (kèm đường đứt nét rule engine 0.5335) + panel phải là Δ zoom quanh 0. |
| `fusion_delta_ci.png` | Hình 2 (forest plot): Δ = fusion − AI kèm CI 95% paired bootstrap - thanh nào cắt trục 0 là không kết luận được; ngôi sao = có ý nghĩa thống kê. |
| `fusion_variants_delta.png` | Hình 3: Δ so với AI-only của cả 4 biến thể fusion (argmax/thresholds/per-channel/meta) cho từng model. |
| `fusion_per_class_f1.png` | Hình 4: F1 từng lớp (HATE/OFFENSIVE/CLEAN) của AI-only vs fusion primary. |
| `fusion_significance.json` | Kiểm định đầy đủ: paired bootstrap (macro-F1) + McNemar (độ đúng) giữa AI-only và fusion `primary`. |
| `significance_report.md` | Bản Markdown của cùng kết quả - đọc file này để biết chênh lệch có ý nghĩa hay không. |
| `<model>/fusion_report.md` | Báo cáo từng model: alpha, per-channel alpha, threshold đóng băng + bảng metric test của 6 hệ thống (AI-only, rule-only, fusion argmax/thresholds/per-channel/meta). |
| `<model>/metrics.json` | Chỉ số đầy đủ của từng biến thể fusion (validation + test) - nguồn dữ liệu của script 29. |
| `<model>/fusion_predictions.csv` | Dự đoán per-sample của biến thể `primary` (nhãn ghi bằng **tên** `HATE/OFFENSIVE/CLEAN`) - đầu vào của script 25. |
| `fusion.log` | Log chạy fusion: device, checkpoint, alpha và threshold đã chọn. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `sensitiveai-vi` | thư mục | 4 file, 1.2 MB |
| `sensitiveai-vi-custom` | thư mục | 4 file, 1.2 MB |
| `sensitiveai-vi-custom-curriculum` | thư mục | 4 file, 1.2 MB |
| `sensitiveai-vi-customdlr2stage` | thư mục | 4 file, 1.2 MB |
| `sensitiveai-vi-customdlr2stage-ext` | thư mục | 4 file, 1.1 MB |
| `sensitiveai-vi-stage2` | thư mục | 4 file, 1.2 MB |
| `comparison.md` | file | 1.3 KB · 2026-10-19 13:14:33 |
| `explain.md` | file | 5.2 KB · 2026-10-19 15:06:21 |
| `fusion.log` | file | 12.8 KB · 2026-10-19 13:14:33 |
| `fusion_ai_vs_fusion.png` | file | 90.0 KB · 2026-10-19 15:04:54 |
| `fusion_delta_ci.png` | file | 68.3 KB · 2026-10-19 15:04:55 |
| `fusion_per_class_f1.png` | file | 100.0 KB · 2026-10-19 15:04:55 |
| `fusion_significance.json` | file | 2.5 KB · 2026-10-19 13:26:05 |
| `fusion_variants_delta.png` | file | 73.1 KB · 2026-10-19 15:04:55 |
| `significance_report.md` | file | 1.6 KB · 2026-10-19 13:26:05 |

## Cách đọc

1. Hình trực quan nhất: `fusion_delta_ci.png` (model nào khác AI thật sự?) rồi `fusion_ai_vs_fusion.png` (số tuyệt đối).
2. Đọc nhanh: `significance_report.md` (chênh lệch có ý nghĩa không?) rồi `comparison.md` (bảng theo model).
3. Δ = AI-only - fusion: Δ dương nghĩa là fusion **kém hơn**; CI chứa 0 = không kết luận được. (Trên hình script 29 đổi chiều: Δ = fusion − AI.)
4. `fusion.log` có dòng `best alpha on validation` và `frozen thresholds` - bằng chứng tham số không lấy từ test.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
