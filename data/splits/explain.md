<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# data/splits - split cố định dùng chung

> **Dùng để làm gì:** **Quan trọng nhất về tính công bằng.** Đây là 3 file CSV mà mọi model đều đọc, kèm SHA-256 của từng file. Mọi lần train/evaluate đều kiểm tra fingerprint trước khi nạp dữ liệu; sai hash là dừng ngay.

> **Sinh ra bởi:** `python scripts/00_prepare_data.py --stage split`

> **Quy tắc giữ lại:** **KHÔNG XÓA, KHÔNG SỬA TAY.** Nếu vô tình đổi, phải tạo lại toàn bộ fingerprint bằng `00_prepare_data.py --stage split --overwrite` - nghĩa là mọi kết quả cũ không còn so sánh được.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `train.csv` | 24.046 mẫu dùng để huấn luyện và ước lượng class weight. |
| `validation.csv` | 2.672 mẫu dùng cho early stopping, chọn best checkpoint và tìm threshold. |
| `test.csv` | 6.680 mẫu dùng để chấm **đúng một lần** ở cuối cùng. |
| `split_manifest.json` | Số mẫu, phân bố nhãn và SHA-256 của từng split + metadata (chiến lược, seed, tỉ lệ). |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `explain.md` | file | 2.1 KB · 2026-10-04 12:58:19 |
| `split_manifest.json` | file | 1.5 KB · 2026-10-04 10:48:01 |
| `test.csv` | file | 533.4 KB · 2026-10-04 10:48:01 |
| `train.csv` | file | 1.9 MB · 2026-10-04 10:48:01 |
| `validation.csv` | file | 210.1 KB · 2026-10-04 10:48:01 |

## Cách đọc

1. Nghi ngờ ai đó đổi dữ liệu: so SHA-256 trong `split_manifest.json` với file trên đĩa.
2. Muốn đổi chiến lược chia: `--set strategy=stratified` rồi chạy lại `00_prepare_data.py --stage split --overwrite` - **lưu `manifest` cũ lại trước**.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:01. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
