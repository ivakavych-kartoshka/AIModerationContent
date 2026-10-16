<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# experiments/<model> - checkpoint của một model

> **Dùng để làm gì:** Một thư mục con `run_XXX` cho mỗi lần chạy.

> **Sinh ra bởi:** `run_experiment`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Danh sách các lần chạy (`run_XXX`)

Cột **Trạng thái**: `đã chấm test` = có `metrics.json` · `đã train` = có `training_summary.json` · `dở dang` = thư mục đã tạo nhưng chưa xong · `chỉ dry-run` = **không phải lần chạy thật** (chỉ có `config.yaml`, không có `training_log.csv`; các thư mục này do `--dry-run` để lại trước khi dry-run được sửa để không ghi file).

| Run | Sửa lần cuối | Số epoch đã log | Best val macro F1 | Trạng thái |
|---|---|---:|---|---|
| `run_001` | 2026-10-11 23:37:30 | 11 | 0.6886 | đã train |
| `run_002` | 2026-10-13 19:21:00 | 15 | 0.6928 | đã train |
| `run_003` | 2026-10-13 14:28:39 | - | - | chỉ dry-run |
| `run_004` | 2026-10-13 14:32:10 | - | - | chỉ dry-run |
| `run_005` | 2026-10-13 14:35:01 | - | - | chỉ dry-run |
| `run_006` | 2026-10-13 14:47:39 | - | - | chỉ dry-run |

Cột *Sửa lần cuối* là thời điểm sửa file trong thư mục run - dùng để biết lần chạy nào mới nhất.

> Bảng này do `scripts/05_document_folders.py` sinh lại. Cần làm mới sau mỗi lần train: `python scripts/05_document_folders.py`

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `run_001` | thư mục | 7 file, 1.0 GB |
| `run_002` | thư mục | 7 file, 1.0 GB |
| `run_003` | thư mục | 2 file, 3.5 KB |
| `run_004` | thư mục | 2 file, 3.5 KB |
| `run_005` | thư mục | 2 file, 3.5 KB |
| `run_006` | thư mục | 2 file, 3.5 KB |
| `explain.md` | file | 2.4 KB · 2026-10-17 00:09:28 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:21. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
