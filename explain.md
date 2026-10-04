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
| `reports/` | Log + artifact của mỗi lần train/evaluate. **Nơu cần tìm log.** |
| `experiments/` | Checkpoint từng stage của mỗi lần train. |
| `checkpoints/` | Kho checkpoint dự trữ. |
| `evaluation/` | Log đánh giá tổng hợp + bảng/biểu đồ so sánh 4 model. |
| `paper/` | Khung bài báo: main.tex, references.bib, sections, tables, figures. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `checkpoints` | thư mục | 1 file, 1.9 KB |
| `configs` | thư mục | 6 file, 16.8 KB |
| `data` | thư mục | 18 file, 7.5 MB |
| `docs` | thư mục | 2 file, 25.2 KB |
| `evaluation` | thư mục | 78 file, 2.3 MB |
| `evaluation_smoke` | thư mục | 18 file, 1.2 MB |
| `experiments` | thư mục | 25 file, 47.7 KB |
| `experiments_smoke` | thư mục | 58 file, 6.2 GB |
| `paper` | thư mục | 18 file, 12.8 KB |
| `paper_smoke` | thư mục | 18 file, 1.1 MB |
| `reports` | thư mục | 55 file, 91.4 KB |
| `reports_smoke` | thư mục | 214 file, 4.2 GB |
| `scripts` | thư mục | 8 file, 20.7 KB |
| `src` | thư mục | 100 file, 905.8 KB |
| `.gitignore` | file | 508 B · 2026-10-04 10:33:39 |
| `explain.md` | file | 3.3 KB · 2026-10-04 12:58:19 |
| `plan.md` | file | 20.2 KB · 2026-10-04 01:32:27 |
| `README.md` | file | 48.1 KB · 2026-10-04 12:15:15 |
| `requirements.txt` | file | 517 B · 2026-10-04 11:31:24 |

## Cách đọc

1. Người mới bắt đầu: đọc `README.md` mục 1 và mục 9.
2. Muốn hiểu code: `src/` (mỗi subfolder có `explain.md` riêng).
3. Muốn xem log train: `reports/<model>/run_XXX/explain.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
