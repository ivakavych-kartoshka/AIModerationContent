<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# experiments/<model> - checkpoint của một model

> **Dùng để làm gì:** Một thư mục con `run_XXX` cho mỗi lần chạy.

> **Sinh ra bởi:** `run_experiment`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Danh sách các lần chạy (`run_XXX`)

Cột **Trạng thái**: `đã chấm test` = có `metrics.json` · `đã train` = có `training_summary.json` · `dở dang` = thư mục đã tạo nhưng chưa xong · `chỉ dry-run` = **không phải lần chạy thật** (chỉ có `config.yaml`, không có `training_log.csv`; các thư mục này do `--dry-run` để lại trước khi dry-run được sửa để không ghi file).

| Run | Sửa lần cuối | Số epoch đã log | Best val macro F1 | Trạng thái |
|---|---|---:|---|---|
| `run_001` | 2026-10-07 03:48:42 | 21 | 0.7089 | đã train |
| `run_002` | 2026-10-10 03:56:12 | 9 | 0.7014 | đã train |
| `run_003` | 2026-10-10 17:58:38 | 9 | 0.6958 | đã train |

Cột *Sửa lần cuối* là thời điểm sửa file trong thư mục run - dùng để biết lần chạy nào mới nhất.

> Bảng này do `scripts/05_document_folders.py` sinh lại. Cần làm mới sau mỗi lần train: `python scripts/05_document_folders.py`

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `run_001` | thư mục | 14 file, 2.1 GB |
| `run_002` | thư mục | 9 file, 1.0 GB |
| `run_003` | thư mục | 9 file, 1.0 GB |
| `explain.md` | file | 2.1 KB · 2026-10-17 00:09:28 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:21. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
