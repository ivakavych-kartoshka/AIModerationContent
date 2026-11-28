<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# configs - cấu hình YAML

> **Dùng để làm gì:** Nguồn sự thật cho mọi tham số thực nghiệm: chia dữ liệu và 4 biến thể mô hình. Mỗi lần train đều lưu bản đã resolve vào `reports/<model>/run_XXX/config.yaml`.

> **Sinh ra bởi:** Chỉnh tay. Đọc bằng `src/common/config.py`; ghi đè tạm thời bằng cờ `--set`.

> **Quy tắc giữ lại:** **KHÔNG SỬA** sau khi đã chạy thật. Muốn thử biến thể mới thì copy sang file mới.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `data_split.yaml` | Nguồn dữ liệu, chiến lược split (official/stratified), tỉ lệ, seed, và **toàn bộ quy tắc làm sạch** văn bản. |
| `sensitiveai-vi.yaml` | Model 1 - baseline: standard `[CLS]` head, cross-entropy, 1 learning rate, 6 epoch, quyết định bằng argmax. |
| `sensitiveai-vi-custom.yaml` | Model 2 - giống baseline, **chỉ thay head** bằng multi-view pooling, 6 epoch. |
| `sensitiveai-vi-customdlr2stage.yaml` | Model 3 - custom head + DLR 4 nhóm + 2 stage (CE -> focal) + class weight + threshold, 7 epoch. |
| `sensitiveai-vi-custom-curriculum.yaml` | Model 4 - custom head + DLR + curriculum 3 stage (OFFENSIVE -> HATE -> CE) + class weight manual + threshold, 10 epoch. |
| `rule_engine.yaml` | Cấu hình rule engine: chuẩn hoá, view chống ngụy trang, ngưỡng khớp và trọng số điểm (plan §37-39). Mọi trọng số là giả thuyết ban đầu, tune trên validation. Baseline hiện tại với lexicon 668 entry: validation macro-F1 0.4555 (thresholds [0.85, 0.0, 0.41]) -> test 0.4536 (đo 1 lần). |
| `rule_engine_tuned.yaml` | Bản tune của rule engine (plan §38-39, §44) do `scripts/27_tune_rule_weights.py` sinh: coordinate descent 155 trial trên **validation** với lexicon 668 entry, đổi 3 trọng số - `match_factors.accentless_alias` 0.90 -> 0.00 (view không dấu khớp nhầm âm tiết phổ biến, plan §38), `match_factors.obfuscated_combined` 0.60 -> 0.45, `negation_factor` 0.50 -> 1.00. Validation macro-F1 0.4555 -> 0.5481; test chấm một lần: 0.5335. Đánh đổi: recall `no_accent` 0.530 -> 0.3625. Không ghi đè bản gốc. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `data_split.yaml` | file | 2.0 KB · 2026-10-04 10:45:52 |
| `explain.md` | file | 3.8 KB · 2026-10-20 07:13:19 |
| `hybrid_fusion.yaml` | file | 2.4 KB · 2026-10-19 13:34:35 |
| `phobert-base.yaml` | file | 2.0 KB · 2026-10-15 15:32:13 |
| `rule_engine.yaml` | file | 2.1 KB · 2026-10-17 01:00:08 |
| `rule_engine_pruned.yaml` | file | 2.1 KB · 2026-10-19 12:38:00 |
| `rule_engine_tuned.yaml` | file | 3.1 KB · 2026-10-19 13:06:04 |
| `rule_engine_tuned_p008.yaml` | file | 3.0 KB · 2026-10-19 12:38:00 |
| `rule_engine_tuned_p010.yaml` | file | 3.0 KB · 2026-10-19 12:38:00 |
| `rule_engine_tuned_p015.yaml` | file | 3.0 KB · 2026-10-19 12:38:00 |
| `sensitiveai-vi-custom-curriculum.yaml` | file | 4.4 KB · 2026-10-06 22:25:01 |
| `sensitiveai-vi-custom.yaml` | file | 2.5 KB · 2026-10-05 22:08:16 |
| `sensitiveai-vi-customdlr2stage-ext.yaml` | file | 3.4 KB · 2026-10-11 12:14:46 |
| `sensitiveai-vi-customdlr2stage.yaml` | file | 4.5 KB · 2026-10-11 12:00:40 |
| `sensitiveai-vi-stage2.yaml` | file | 2.9 KB · 2026-10-13 14:30:37 |
| `sensitiveai-vi.yaml` | file | 1.9 KB · 2026-10-05 22:07:41 |
| `visobert-base.yaml` | file | 2.0 KB · 2026-10-15 15:32:24 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
