<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# experiments - checkpoint từng stage

> **Dùng để làm gì:** Tách khỏi `reports/` để phân biệt **log/kết quả** với **trọng số mô hình**. Mỗi lần chạy có `experiments/<model>/run_XXX/stage_checkpoints/<stage>/` - cần cho multi-stage (model 3, 4) vì stage sau nạp checkpoint của stage trước.

> **Sinh ra bởi:** `python -m src.training.run_experiment --config <model>`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

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
| `sensitiveai-vi` | thư mục | 4 file, 12.3 KB |
| `sensitiveai-vi-custom` | thư mục | 4 file, 12.5 KB |
| `sensitiveai-vi-custom-curriculum` | thư mục | 32 file, 4.2 GB |
| `sensitiveai-vi-customdlr2stage` | thư mục | 17 file, 2.1 GB |
| `explain.md` | file | 2.2 KB · 2026-10-04 12:58:19 |

## Cách đọc

1. Muốn nạp lại một stage giữa chừng để debug: `--checkpoint experiments/<model>/run_XXX/stage_checkpoints/stage_1`.
2. Muốn biết stage nào tồn tại: xem `stage_transitions.json` trong `reports/<model>/run_XXX/`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
