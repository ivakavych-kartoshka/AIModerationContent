<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# reports/<model> - tất cả lần chạy của một model

> **Dùng để làm gì:** Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập.

> **Sinh ra bởi:** `run_experiment` tự cấp số thứ tự tự do kế tiếp.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `run_pointer.txt` | Tên thư mục của lần chạy mới nhất. |
| `explain.md` | File này, kèm bảng liệt kê các lần chạy. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `best_model` | thư mục | 8 file, 1.1 GB |
| `last_model` | thư mục | 8 file, 1.1 GB |
| `best_training_state.pt` | file | 2.1 GB · 2026-10-10 02:49:16 |
| `class_weights.json` | file | 577 B · 2026-10-10 02:15:45 |
| `config.yaml` | file | 2.9 KB · 2026-10-10 02:15:45 |
| `explain.md` | file | 2.0 KB · 2026-10-19 15:06:21 |
| `split_manifest_used.json` | file | 397 B · 2026-10-10 02:15:45 |
| `stage_transitions.json` | file | 1.9 KB · 2026-10-10 03:56:08 |
| `train.log` | file | 3.6 KB · 2026-10-10 03:56:12 |
| `training_log.csv` | file | 2.4 KB · 2026-10-10 03:56:07 |
| `training_summary.json` | file | 9.7 KB · 2026-10-10 03:56:12 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:40. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
