<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# data - dữ liệu ViHSD

> **Dùng để làm gì:** Ba lớp dữ liệu: bản tải về nguyên bản, bản đã làm sạch, và bản split cố định dùng chung cho mọi model. Chỉ tầng `splits/` là thứ mô hình thực sự đọc.

> **Sinh ra bởi:** `python scripts/00_prepare_data.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `raw/` | Dữ liệu tải về, không sửa. |
| `processed/` | Sau làm sạch và remap nhãn. |
| `splits/` | Split cố định + SHA-256. **Đây là nguồn sự thật cho mọi lần chạy.** |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `processed` | thư mục | 6 file, 2.7 MB |
| `raw` | thư mục | 6 file, 2.1 MB |
| `splits` | thư mục | 5 file, 2.7 MB |
| `explain.md` | file | 1.7 KB · 2026-10-04 12:58:19 |

## Cách đọc

1. Muốn biết dữ liệu đã qua xử lý gì: `processed/vihsd/preprocessing_report.json`.
2. Muốn xác nhận 4 model dùng cùng dữ liệu: `splits/split_manifest.json` (SHA-256 từng file).

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
