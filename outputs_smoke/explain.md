<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# outputs_smoke - kết quả chạy kiểm thử nhanh (smoke test)

> **Dùng để làm gì:** Thư mục chứa toàn bộ kết quả của các lần chạy smoke test kiểm thử pipeline: `reports/`, `experiments/`, `evaluation/`, và `paper/`.

> **Sinh ra bởi:** Các lệnh chạy với cờ `--smoke-test` hoặc trỏ `--reports-dir outputs_smoke/reports`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `reports/` | Báo cáo chi tiết các lần chạy smoke test. |
| `experiments/` | Trạng thái checkpoint smoke test. |
| `evaluation/` | Đánh giá tổng hợp smoke test. |
| `paper/` | Bảng số liệu smoke test. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `evaluation_smoke` | thư mục | 97 file, 11.6 MB |
| `experiments_smoke` | thư mục | 92 file, 10.4 GB |
| `reports_smoke` | thư mục | 429 file, 9.5 GB |
| `explain.md` | file | 1.9 KB · 2026-10-17 00:09:27 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
