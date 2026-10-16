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
| `rule_engine.yaml` | Cấu hình rule engine: chuẩn hoá, view chống ngụy trang, ngưỡng khớp và trọng số điểm (plan §37-39). Mọi trọng số là giả thuyết ban đầu, tune trên validation. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `data_split.yaml` | file | 2.0 KB · 2026-10-04 10:45:52 |
| `explain.md` | file | 2.7 KB · 2026-10-17 00:09:27 |
| `phobert-base.yaml` | file | 2.0 KB · 2026-10-15 15:32:13 |
| `rule_engine.yaml` | file | 2.1 KB · 2026-10-16 23:35:44 |
| `sensitiveai-vi-custom-curriculum.yaml` | file | 4.4 KB · 2026-10-06 22:25:01 |
| `sensitiveai-vi-custom.yaml` | file | 2.5 KB · 2026-10-05 22:08:16 |
| `sensitiveai-vi-customdlr2stage-ext.yaml` | file | 3.4 KB · 2026-10-11 12:14:46 |
| `sensitiveai-vi-customdlr2stage.yaml` | file | 4.5 KB · 2026-10-11 12:00:40 |
| `sensitiveai-vi-stage2.yaml` | file | 2.9 KB · 2026-10-13 14:30:37 |
| `sensitiveai-vi.yaml` | file | 1.9 KB · 2026-10-05 22:07:41 |
| `visobert-base.yaml` | file | 2.0 KB · 2026-10-15 15:32:24 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
