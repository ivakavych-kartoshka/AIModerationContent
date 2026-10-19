<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# src/fusion - mã nguồn hybrid fusion (rule engine + AI)

> **Dùng để làm gì:** Ghép điểm rule engine với xác suất AI (plan §40-41): quét `alpha` (trọng số AI), per-channel alpha, tune threshold và huấn luyện meta-classifier - **tất cả trên validation**, đóng băng rồi chấm test một lần (plan §44). Không tự chứa tham số: đọc `configs/hybrid_fusion.yaml`.

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Chạy qua `python scripts/23_evaluate_fusion.py`; kết quả ở `evaluation/fusion/`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `run_fusion.py` | Điều phối toàn bộ: tìm run/checkpoint AI, dùng lại `per_model/<model>/{validation,test}_predictions.csv`, ghép với `rule_scores.csv`, chọn alpha/threshold/meta trên validation, ghi báo cáo + `fusion_significance` bước sau. |
| `weighted_fusion.py` | Hàm thuần: `score_alpha_grid`, per-channel alpha, `tune_thresholds_at_alpha(s)`, `MetaFusion` (logistic regression trên 6 điểm rule/AI). |
| `__init__.py` | Export công khai của package. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 3 file, 61.4 KB |
| `__init__.py` | file | 824 B · 2026-10-19 07:09:20 |
| `explain.md` | file | 2.4 KB · 2026-10-19 15:06:20 |
| `run_fusion.py` | file | 29.5 KB · 2026-10-19 07:10:33 |
| `weighted_fusion.py` | file | 11.9 KB · 2026-10-19 07:09:11 |

## Cách đọc

1. Muốn hiểu 'alpha' ảnh hưởng gì: đọc `score_alpha_grid` trong `weighted_fusion.py` (1.0 = chỉ AI, 0.0 = chỉ rule).
2. Kết quả và kiểm định: `evaluation/fusion/significance_report.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:39. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
