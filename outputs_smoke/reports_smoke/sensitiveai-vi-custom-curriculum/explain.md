<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# reports/<model> - tất cả lần chạy của một model

> **Dùng để làm gì:** Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập.

> **Sinh ra bởi:** `run_experiment` tự cấp số thứ tự tự do kế tiếp.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `run_pointer.txt` | Tên thư mục của lần chạy mới nhất. |
| `explain.md` | File này, kèm bảng liệt kê các lần chạy. |

## Danh sách các lần chạy (`run_XXX`)

Cột **Trạng thái**: `đã chấm test` = có `metrics.json` · `đã train` = có `training_summary.json` · `dở dang` = thư mục đã tạo nhưng chưa xong · `chỉ dry-run` = **không phải lần chạy thật** (chỉ có `config.yaml`, không có `training_log.csv`; các thư mục này do `--dry-run` để lại trước khi dry-run được sửa để không ghi file).

| Run | Sửa lần cuối | Số epoch đã log | Best val macro F1 | Trạng thái |
|---|---|---:|---|---|
| `run_001` | 2026-10-04 12:47:43 | 1 | - | dở dang |
| `run_002` | 2026-10-04 12:51:27 | 3 | - | đã chấm test |

Cột *Sửa lần cuối* là thời điểm sửa file trong thư mục run - dùng để biết lần chạy nào mới nhất.

> Bảng này do `scripts/05_document_folders.py` sinh lại. Cần làm mới sau mỗi lần train: `python scripts/05_document_folders.py`
## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `run_001` | thư mục | 7 file, 13.9 KB |
| `run_002` | thư mục | 37 file, 1.1 GB |
| `explain.md` | file | 3.1 KB · 2026-10-04 12:58:19 |
| `run_pointer.txt` | file | 7 B · 2026-10-04 12:50:49 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
