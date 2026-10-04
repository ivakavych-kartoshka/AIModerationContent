<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# training - quy trình huấn luyện

> **Dùng để làm gì:** Dựng optimizer theo nhóm DLR, lập kế hoạch stage, chạy vòng lặp train tùy chỉnh (không dùng Hugging Face `Trainer` vì API đổi ở transformers 5.x) và cung cấp CLI.

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Kết quả nằm ở `reports/` và `experiments/`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `optim.py` | Chia tham số thành 4 nhóm (embeddings / encoder LN+rel / block dưới / block trên / head), tách weight decay cho bias-LayerNorm, dựng scheduler. |
| `stages.py` | `resolve_stages()` - merge cấu hình global với override theo từng stage. |
| `trainer.py` | Vòng lặp train: gradient accumulation, clip grad norm, evaluate mỗi epoch, callback, lưu `last_model`, viết `training_summary.json`. |
| `run_experiment.py` | CLI train một model. Tự cấp `run_XXX` kế tiếp, ghi `train.log`, và **từ chối ghi đè** một run đã có dữ liệu. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 5 file, 82.0 KB |
| `__init__.py` | file | 575 B · 2026-10-04 10:41:56 |
| `explain.md` | file | 2.1 KB · 2026-10-04 12:58:19 |
| `optim.py` | file | 10.7 KB · 2026-10-04 11:39:41 |
| `run_experiment.py` | file | 15.9 KB · 2026-10-04 12:49:03 |
| `stages.py` | file | 6.6 KB · 2026-10-04 10:50:09 |
| `trainer.py` | file | 23.3 KB · 2026-10-04 11:39:41 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
