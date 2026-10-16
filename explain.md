<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# AIModerationContent - SensitiveAI

> **Dùng để làm gì:** Thư mục gốc của dự án phân loại ngôn tữ độc hại tiếng Việt (HATE / OFFENSIVE / CLEAN) trên ViHSD, so sánh 4 hệ thống dùng chung backbone `microsoft/mdeberta-v3-base`.

`README.md` là tài liệu chính để mentor đọc. File này (`explain.md`) chỉ giải thích cấu trúc thư mục.

> **Sinh ra bởi:** Không sinh ra - đây là thư mục gốc của kho mã.

> **Quy tắc giữ lại:** Thư mục gốc không được xóa. Các thư mục con sinh ra khi chạy có quy tắc giữ riêng.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `README.md` | Tài liệu chi tiết toàn diện: bài toán, 4 hệ thống, dữ liệu, chiến lược train, lệnh chạy, cấu trúc output, quy chuẩn công bằng, khắc phục sự cố. |
| `plan.md` | Kế hoạch nghiên cứu gốc. Mọi quyết định thiết kế trong mã đều truy về đây. |
| `requirements.txt` | Dependency và phiên bản tối thiểu. |
| `explain.md` | File này. |

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `configs/` | 5 file YAML: data split + 4 cấu hình model. |
| `src/` | Mã nguồn Python (43 module). |
| `scripts/` | 5 entry point chạy theo thứ tự 00 -> 05. |
| `data/` | Dữ liệu ViHSD: raw, processed, splits. |
| `outputs/` | Toàn bộ kết quả chính thức: reports, experiments, evaluation, paper. |
| `outputs_smoke/` | Toàn bộ kết quả smoke test (nếu gom vào thư mục con). |
| `checkpoints/` | Kho checkpoint dự trữ. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `.freebuff` | thư mục | 1 file, 37 B |
| `.git` | thư mục | 1652 file, 691.5 MB |
| `checkpoints` | thư mục | 1 file, 1.3 KB |
| `configs` | thư mục | 10 file, 27.9 KB |
| `data` | thư mục | 33 file, 8.0 MB |
| `docs` | thư mục | 2 file, 25.2 KB |
| `outputs` | thư mục | 633 file, 57.5 GB |
| `outputs_smoke` | thư mục | 612 file, 19.9 GB |
| `papers` | thư mục | 127 file, 14.1 MB |
| `scripts` | thư mục | 17 file, 146.5 KB |
| `src` | thư mục | 104 file, 1.0 MB |
| `.gitignore` | file | 848 B · 2026-10-16 16:01:27 |
| `explain.md` | file | 3.3 KB · 2026-10-04 13:00:00 |
| `fix.txt` | file | 15 B · 2026-10-15 19:35:21 |
| `plan.md` | file | 20.1 KB · 2026-10-04 13:04:21 |
| `README.md` | file | 48.7 KB · 2026-10-05 22:28:54 |
| `requirements.txt` | file | 517 B · 2026-10-04 11:31:24 |
| `texput.log` | file | 744 B · 2026-10-15 21:33:57 |

## Cách đọc

1. Người mới bắt đầu: đọc `README.md` mục 1 và mục 9.
2. Muốn hiểu code: `src/` (mỗi subfolder có `explain.md` riêng).
3. Muốn xem log train: `outputs/reports/<model>/run_XXX/explain.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-16 16:02:28. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
