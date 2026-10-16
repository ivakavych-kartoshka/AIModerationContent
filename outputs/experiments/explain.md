<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# experiments - checkpoint từng stage

> **Dùng để làm gì:** Tách khỏi `reports/` để phân biệt **log/kết quả** với **trọng số mô hình**. Mỗi lần chạy có `experiments/<model>/run_XXX/stage_checkpoints/<stage>/` - cần cho multi-stage (model 3, 4) vì stage sau nạp checkpoint của stage trước.

> **Sinh ra bởi:** `python -m src.training.run_experiment --config <model>`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `sensitiveai-vi/` | Model 1 (1 stage). |
| `sensitiveai-vi-custom/` | Model 2 (1 stage). |
| `sensitiveai-vi-customdlr2stage/` | Model 3 (2 stage). |
| `sensitiveai-vi-custom-curriculum/` | Model 4 (3 stage). |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `sensitiveai-vi` | thư mục | 4 file, 15.1 KB |
| `sensitiveai-vi-custom` | thư mục | 4 file, 16.5 KB |
| `sensitiveai-vi-custom-curriculum` | thư mục | 20 file, 3.1 GB |
| `sensitiveai-vi-customdlr2stage` | thư mục | 33 file, 4.2 GB |
| `sensitiveai-vi-customdlr2stage-ext` | thư mục | 12 file, 1.0 GB |
| `sensitiveai-vi-stage2` | thư mục | 23 file, 2.1 GB |
| `explain.md` | file | 2.1 KB · 2026-10-17 00:09:27 |

## Cách đọc

1. Muốn nạp lại một stage giữa chừng để debug: `--checkpoint experiments/<model>/run_XXX/stage_checkpoints/stage_1`.
2. Muốn biết stage nào tồn tại: xem `stage_transitions.json` trong `reports/<model>/run_XXX/`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
