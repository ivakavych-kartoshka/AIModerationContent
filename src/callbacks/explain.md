<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# callbacks - callback huấn luyện

> **Dùng để làm gì:** Các bước gắn vào vòng lặp train: ghi CSV từng epoch, lưu best checkpoint, early stopping, điều phối chuyển stage.

> **Sinh ra bởi:** Không sinh ra - mã nguồn.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `csv_logger.py` | `CSVLoggerCallback` - ghi `training_log.csv`, **một dòng mỗi epoch**, có cột `is_best`. Không vẽ lại từ RAM. |
| `monitoring.py` | `BestCheckpointCallback` + `EarlyStoppingCallback` - chọn epoch tốt nhất theo **validation** macro F1. |
| `stage_controller.py` | Điều phối stage: nạp `init_from`, ghi checkpoint stage, ghi `stage_transitions.json`. |
| `base.py` | Giao diện callback chung. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 5 file, 37.7 KB |
| `__init__.py` | file | 638 B · 2026-10-04 10:40:30 |
| `base.py` | file | 3.9 KB · 2026-10-04 10:39:47 |
| `csv_logger.py` | file | 6.4 KB · 2026-10-04 10:40:02 |
| `explain.md` | file | 1.9 KB · 2026-10-04 12:58:19 |
| `monitoring.py` | file | 6.3 KB · 2026-10-04 11:05:29 |
| `stage_controller.py` | file | 6.6 KB · 2026-10-04 12:38:57 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
