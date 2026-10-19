<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# experiments/<model> - checkpoint của một model

> **Dùng để làm gì:** Một thư mục con `run_XXX` cho mỗi lần chạy.

> **Sinh ra bởi:** `run_experiment`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Danh sách các lần chạy (`run_XXX`)

Cột **Trạng thái**: `đã chấm test` = có `metrics.json` · `đã train` = có `training_summary.json` · `dở dang` = thư mục đã tạo nhưng chưa xong · `chỉ dry-run` = **không phải lần chạy thật** (chỉ có `config.yaml`, không có `training_log.csv`; các thư mục này do `--dry-run` để lại trước khi dry-run được sửa để không ghi file).

| Run | Sửa lần cuối | Số epoch đã log | Best val macro F1 | Trạng thái |
|---|---|---:|---|---|
| `run_001` | 2026-10-17 01:00:08 | 19 | 0.6942 | đã train |

Cột *Sửa lần cuối* là thời điểm sửa file trong thư mục run - dùng để biết lần chạy nào mới nhất.

> Bảng này do `scripts/05_document_folders.py` sinh lại. Cần làm mới sau mỗi lần train: `python scripts/05_document_folders.py`

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `run_001` | thư mục | 19 file, 3.1 GB |
| `explain.md` | file | 1.8 KB · 2026-10-19 15:06:21 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
