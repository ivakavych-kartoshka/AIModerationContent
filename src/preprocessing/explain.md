<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# preprocessing - làm sạch văn bản

> **Dùng để làm gì:** Chuẩn hoá văn bản tiếng Việt trước khi mã hoá, theo nguyên tắc **không thao tác nào xóa thông tin được bật mặc định**: mọi tuỳ chọn nguy hiểm (xoá URL, xoá mention, gộp dấu câu lặp) đều tắt mặc định.

> **Sinh ra bởi:** Không sinh ra - mã nguồn.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `text_cleaning.py` | 12 bước: hạ chữ Unicode, NFC, giải thực thể HTML, bỏ ký tự vô hình, bỏ thẻ HTML, chuẩn hoá space, sửa khoảng trắng quanh dấu câu, gộp space, cắt đầu/cuối. Chỉ loại dòng mà **không còn ký tự nhìn thấy nào**. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 2 file, 11.1 KB |
| `__init__.py` | file | 380 B · 2026-10-04 10:35:08 |
| `explain.md` | file | 1.7 KB · 2026-10-04 12:58:19 |
| `text_cleaning.py` | file | 7.9 KB · 2026-10-04 10:47:54 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
