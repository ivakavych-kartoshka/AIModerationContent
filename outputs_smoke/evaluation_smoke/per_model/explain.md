<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/per_model - số liệu theo từng model

> **Dùng để làm gì:** Gom chỉ số chính của từng model ra một chỗ, tiện để so sánh nhanh mà không phải mở từng `run_XXX`.

> **Sinh ra bởi:** `run_evaluation` (`--output-dir` hoặc mặc định ghi vào đây).

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `<model>/` | Số liệu của một model. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `sensitiveai-vi` | thư mục | 18 file, 2.5 MB |
| `sensitiveai-vi-custom` | thư mục | 18 file, 2.6 MB |
| `sensitiveai-vi-custom-curriculum` | thư mục | 19 file, 2.5 MB |
| `sensitiveai-vi-customdlr2stage` | thư mục | 19 file, 2.7 MB |
| `explain.md` | file | 1.7 KB · 2026-10-17 00:09:27 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
