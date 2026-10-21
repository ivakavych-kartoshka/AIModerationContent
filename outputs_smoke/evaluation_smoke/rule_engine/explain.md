<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# outputs_smoke/evaluation_smoke/rule_engine - smoke test rule engine

> **Dùng để làm gì:** Kết quả **smoke test** nhanh cho rule engine từ vựng (không cần GPU, chạy vài giây). Chỉ kiểm tra code chạy đúng (offset map, giữ dấu, khớp exact, phân loại biến đổi, xử lý text rỗng/rất dài) và chạy end-to-end trên một phần hoặc **toàn bộ ViHSD** bằng **argmax trực tiếp, KHÔNG tune threshold** nên số liệu thấp hơn `evaluation/rule_engine/` (nơi có tune threshold).

> **Sinh ra bởi:** `python scripts/18_smoke_test_rule_engine.py --split all --limit 0`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `smoke_test.json` | Kết quả đầy đủ dạng JSON: danh sách check (label/ok/detail), metrics end-to-end theo từng split + combined, meta (config, lexicon, split, limit) và cờ `passed`. |
| `smoke_test.md` | Bản báo cáo Markdown của cùng nội dung (bảng check + bảng metrics) để đọc nhanh. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `explain.md` | file | 2.5 KB · 2026-10-20 07:13:19 |
| `smoke_test.json` | file | 3.4 KB · 2026-10-19 14:29:26 |
| `smoke_test.md` | file | 2.1 KB · 2026-10-19 14:29:26 |

## Cách đọc

1. Xem nhanh: `smoke_test.md` (bảng check và bảng macro-F1/accuracy theo split).
2. Muốn trích xuất: đọc `smoke_test.json` (khóa `passed`, `eval`, `checks`).
3. Số liệu chính thức (có tune threshold): dùng `evaluation/rule_engine/`, không dùng folder này.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:44. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
