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

## Danh sách các lần chạy (`run_XXX`)

Cột **Trạng thái**: `đã chấm test` = có `metrics.json` · `đã train` = có `training_summary.json` · `dở dang` = thư mục đã tạo nhưng chưa xong · `chỉ dry-run` = **không phải lần chạy thật** (chỉ có `config.yaml`, không có `training_log.csv`; các thư mục này do `--dry-run` để lại trước khi dry-run được sửa để không ghi file).

| Run | Sửa lần cuối | Số epoch đã log | Best val macro F1 | Trạng thái |
|---|---|---:|---|---|
| `run_001` | 2026-10-04 12:44:25 | - | - | chỉ dry-run |
| `run_002` | 2026-10-04 12:44:25 | - | - | chỉ dry-run |
| `run_003` | 2026-10-04 12:44:25 | - | - | chỉ dry-run |
| `run_004` | 2026-10-04 12:44:25 | - | - | chỉ dry-run |

Cột *Sửa lần cuối* là thời điểm sửa file trong thư mục run - dùng để biết lần chạy nào mới nhất.

> Bảng này do `scripts/05_document_folders.py` sinh lại. Cần làm mới sau mỗi lần train: `python scripts/05_document_folders.py`
## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `run_001` | thư mục | 5 file, 6.9 KB |
| `run_002` | thư mục | 5 file, 6.9 KB |
| `run_003` | thư mục | 5 file, 6.9 KB |
| `run_004` | thư mục | 5 file, 6.9 KB |
| `explain.md` | file | 3.0 KB · 2026-10-04 12:58:19 |

## Cách đọc

1. So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._



Log chi tiết được ghi lúc chạy, theo từng epoch, không cần phải chạy lại. Cụ thể:

1. Log console từng epoch →  train.log 

- Mỗi epoch, log chi tiết ra console (loss, val_loss, val_acc, val_macro_f1, thời gian) và đồng thời ghi vào  reports/<model>/run_XXX/train.log .
- File  train.log  giữ log đầy đủ của cả run: mọi epoch, mọi stage, cảnh báo, plan, best checkpoint, v.v.

2. Log mỗi epoch →  training_log.csv 

-  CSVLoggerCallback  ghi một dòng mỗi epoch vào  training_log.csv  (mở file  w  header lần đầu, sau đó  a  theo epoch).
- Cột bao gồm:  epoch ,  stage ,  loss ,  val_loss ,  val_accuracy ,  val_macro_f1 ,  grad_norm ,  epoch_seconds ,  cumulative_seconds ,  is_best ...
- Đây là file chính để vẽ biểu đồ validation, không phụ thuộc vào memory.

3. Log của từng stage →  stage_transitions.json 

- Mỗi khi chuyển stage,  StageController  ghi  stage_begin / stage_end  vào  stage_transitions.json  (audit trail multi-stage).

4. Log riêng stage = gì?

- Theo code, các stage chia epochs khác nhau, log của mỗi stage được ghi vào cùng  training_log.csv  với cột  stage  +  epoch_in_stage  phân biệt.
-  stage_transitions.json  ghi chi tiết sự kiện chuyển stage.

5. Nếu train bị lỗi/ngắt giữa chừng?

- Nếu crash không gọi  on_train_end ,  train.log  và  training_log.csv  vẫn giữ những dòng đã ghi trước đó (append). Sẽ còn thiếu phần cuối, nhưng không bị ghi đè.
- Nếu chạy lại cùng một  run_id  (không  --allow-run-reuse ), code từ chối ghi đè (chặn FileExistsError). Vậy nên chạy xong không nên cài lại cùng run.

