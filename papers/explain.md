<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# paper - khung bài báo

> **Dùng để làm gì:** Khung bài báo LaTeX. Bảng và hình được **sinh tự động** từ kết quả thực nghiệm, không gõ tay.

> **Sinh ra bởi:** `python scripts/03_compare_models.py` (sinh bảng + hình); `main.tex` và `references.bib` viết tay.

> **Quy tắc giữ lại:** **GIỮ NGUYÊN `main.tex`, `references.bib` và `sections/`.** Riêng `tables/` và `figures/` được sinh lại mỗi lần chạy `03_compare_models.py`.

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
| `figures` | thư mục | 19 file, 3.6 MB |
| `sections` | thư mục | 11 file, 36.1 KB |
| `supplementary` | thư mục | 87 file, 10.8 MB |
| `tables` | thư mục | 11 file, 12.9 KB |
| `template` | thư mục | 8 file, 666.8 KB |
| `explain.md` | file | 2.6 KB · 2026-10-20 07:13:19 |
| `main.aux` | file | 16.2 KB · 2026-10-15 23:05:26 |
| `main.bbl` | file | 5.7 KB · 2026-10-15 23:05:25 |
| `main.blg` | file | 1.0 KB · 2026-10-15 23:05:25 |
| `main.fdb_latexmk` | file | 29.9 KB · 2026-10-15 23:05:27 |
| `main.fls` | file | 34.1 KB · 2026-10-15 23:05:26 |
| `main.log` | file | 34.7 KB · 2026-10-15 23:05:26 |
| `main.out` | file | 531 B · 2026-10-15 23:05:26 |
| `main.pdf` | file | 485.1 KB · 2026-10-15 23:05:26 |
| `main.synctex.gz` | file | 131.0 KB · 2026-10-15 23:05:26 |
| `main.tex` | file | 4.4 KB · 2026-10-15 23:04:52 |
| `main.txt` | file | 41.8 KB · 2026-10-15 01:40:24 |
| `references.bib` | file | 7.5 KB · 2026-10-15 01:18:40 |
| `SUBMISSION_GUIDE.md` | file | 2.7 KB · 2026-10-14 22:29:33 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
