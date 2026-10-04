<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# src - mã nguồn

> **Dùng để làm gì:** Toàn bộ logic của dự án, chia theo vai trò. Không có file nào ở đây được sinh ra khi chạy, và không có script nào ghi vào thư mục này.

> **Sinh ra bởi:** Không sinh ra - mã nguồn.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Sửa mã ở đây không ảnh hưởng tới log đã tạo.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `explain.md` | File này. |

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `common/` | Hằng số nhãn, nạp config, quy ước đường dẫn, seed, logging, thông tin môi trường, và registry `explain.md`. |
| `preprocessing/` | Làm sạch văn bản (12 bước, không xóa thông tin). |
| `data/` | Tải, tiền xử lý, tách split có fingerprint, và dataset đã tokenize. |
| `models/` | Bọc encoder + standard head / custom multi-view head, lưu & nạp checkpoint. |
| `losses/` | Cross-entropy, focal loss, class weight, concept-focused curriculum loss. |
| `callbacks/` | Ghi CSV từng epoch, theo dõi best checkpoint, early stopping, điều phối stage. |
| `training/` | Optimizer theo nhóm DLR, kế hoạch stage, vòng lặp train, CLI `run_experiment`. |
| `evaluation/` | Chỉ số, đường cong, tìm threshold, benchmark tốc độ, biểu đồ, CLI đánh giá. |
| `comparison/` | Gộp 4 model và sinh bảng + LaTeX cho paper. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 1 file, 272 B |
| `callbacks` | thư mục | 11 file, 63.4 KB |
| `common` | thư mục | 17 file, 190.4 KB |
| `comparison` | thư mục | 5 file, 71.3 KB |
| `data` | thư mục | 13 file, 98.4 KB |
| `evaluation` | thư mục | 17 file, 199.5 KB |
| `losses` | thư mục | 11 file, 59.4 KB |
| `models` | thư mục | 7 file, 58.5 KB |
| `preprocessing` | thư mục | 5 file, 21.1 KB |
| `training` | thư mục | 11 file, 141.1 KB |
| `__init__.py` | file | 111 B · 2026-10-04 10:36:17 |
| `explain.md` | file | 2.4 KB · 2026-10-04 12:58:19 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
