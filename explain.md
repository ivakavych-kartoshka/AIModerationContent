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
| `configs/` | YAML: data split + 4 cấu hình model + 2 cấu hình rule engine. |
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
| `.git` | thư mục | 3520 file, 707.2 MB |
| `.kilo` | thư mục | 2 file, 1.5 KB |
| `checkpoints` | thư mục | 1 file, 1.3 KB |
| `configs` | thư mục | 17 file, 48.3 KB |
| `data` | thư mục | 73 file, 18.5 MB |
| `docs` | thư mục | 4 file, 59.1 KB |
| `outputs` | thư mục | 825 file, 57.5 GB |
| `outputs_smoke` | thư mục | 620 file, 19.9 GB |
| `papers` | thư mục | 150 file, 15.8 MB |
| `scripts` | thư mục | 44 file, 506.0 KB |
| `src` | thư mục | 144 file, 1.8 MB |
| `.gitignore` | file | 857 B · 2026-10-19 07:42:22 |
| `explain.md` | file | 3.6 KB · 2026-10-20 07:13:19 |
| `plan.md` | file | 20.1 KB · 2026-10-19 14:15:48 |
| `README.md` | file | 48.7 KB · 2026-10-05 22:28:54 |
| `requirements.txt` | file | 517 B · 2026-10-04 11:31:24 |
| `Rule_Engine_Plan.md` | file | 11.7 KB · 2026-10-17 01:00:08 |
| `tmp_ablation_table.py` | file | 1.3 KB · 2026-10-19 10:17:11 |
| `tmp_context_fp.py` | file | 2.7 KB · 2026-10-19 13:08:02 |
| `tmp_hashes.py` | file | 1.8 KB · 2026-10-19 12:37:36 |
| `tmp_labelmix.py` | file | 835 B · 2026-10-19 12:37:51 |
| `tmp_make_configs.py` | file | 1.2 KB · 2026-10-19 12:37:51 |
| `tmp_prune_sweep.py` | file | 2.1 KB · 2026-10-19 12:37:36 |
| `tmp_report_lexicon.py` | file | 1.0 KB · 2026-10-19 09:18:55 |
| `tmp_summarize17.py` | file | 1.8 KB · 2026-10-19 09:48:08 |

## Cách đọc

1. Người mới bắt đầu: đọc `README.md` mục 1 và mục 9.
2. Muốn hiểu code: `src/` (mỗi subfolder có `explain.md` riêng).
3. Muốn xem log train: `outputs/reports/<model>/run_XXX/explain.md`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
