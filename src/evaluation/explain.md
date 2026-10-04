<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation - đánh giá

> **Dùng để làm gì:** Chấm điểm model đã huấn luyện. Nguyên tắc bất di bất dịch: **chỉ chấm test đúng một lần**, tìm threshold **trên validation**, dùng class weight đã đóng băng từ lúc train.

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Kết quả nằm trong `reports/<model>/run_XXX/` và `evaluation/`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `metrics.py` | Accuracy, macro/weighted F1, per-class P/R/F1, macro & per-class ROC-AUC và average precision, ma trận nhầm lẫn. |
| `thresholds.py` | Tìm threshold riêng từng nhãn bằng **coordinate ascent trên validation**, mục tiêu mặc định macro F1. |
| `benchmark.py` | Đo latency (ms/sample) và throughput (samples/s) với warmup, không tính thời gian warmup vào kết quả. |
| `curves.py` | Chuỗi dữ liệu cho PR/ROC curve. |
| `plots.py` | Vẽ ma trận nhầm lẫn, PR/ROC, per-class F1, đường cong validation theo epoch. |
| `report.py` | `model_summary.txt` và khối thông tin môi trường để tái lập. |
| `run_evaluation.py` | CLI đánh giá một model; **lưu trữ** kết quả cũ vào `history/` thay vì xóa. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 8 file, 120.2 KB |
| `__init__.py` | file | 879 B · 2026-10-04 10:44:02 |
| `benchmark.py` | file | 5.4 KB · 2026-10-04 10:43:04 |
| `curves.py` | file | 5.3 KB · 2026-10-04 11:40:09 |
| `explain.md` | file | 2.5 KB · 2026-10-04 12:58:19 |
| `metrics.py` | file | 12.6 KB · 2026-10-04 11:12:26 |
| `plots.py` | file | 12.1 KB · 2026-10-04 11:39:41 |
| `report.py` | file | 8.3 KB · 2026-10-04 11:39:41 |
| `run_evaluation.py` | file | 26.2 KB · 2026-10-04 12:38:31 |
| `thresholds.py` | file | 6.1 KB · 2026-10-04 10:42:08 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
