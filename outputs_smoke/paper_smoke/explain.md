<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# paper - khung bài báo

> **Dùng để làm gì:** Khung bài báo LaTeX. Bảng và hình được **sinh tự động** từ kết quả thực nghiệm, không gõ tay.

> **Sinh ra bởi:** `python scripts/03_compare_models.py` (sinh bảng + hình); `main.tex` và `references.bib` viết tay.

> **Quy tắc giữ lại:** **GIỮ NGUYÊN `main.tex`, `references.bib` và `sections/`.** Riêng `tables/` và `figures/` được sinh lại mỗi lần chạy `03_compare_models.py`.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `main.tex` | File chính: tiêu đề, tác giả, abstract, các mục, và lệnh `\input` các bảng/hình đã sinh. |
| `references.bib` | 13 tài liệu tham khảo (ViHSD, mDeBERTa-v3, focal loss, curriculum, PR vs ROC, ...). |
| `explain.md` | File này. |

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `sections/` | Các mục của bài báo - hiện là placeholder, cần điền sau khi có kết quả thật. |
| `tables/` | Bảng LaTeX **sinh tự động**. Sửa tay sẽ bị ghi đè lần `03` kế tiếp. |
| `figures/` | Hình PNG **copy tự động** từ `evaluation/comparison/`. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `figures` | thư mục | 13 file, 1.1 MB |
| `tables` | thư mục | 4 file, 4.8 KB |
| `explain.md` | file | 2.0 KB · 2026-10-04 12:58:19 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
