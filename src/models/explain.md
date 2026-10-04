<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# models - kiến trúc mô hình

> **Dùng để làm gì:** Bọc `microsoft/mdeberta-v3-base` với hai loại head: standard `[CLS]` (model 1) và custom multi-view cls+mean+max (model 2/3/4). Cùng một đường nạp/lưu cho cả hai để so sánh công bằng.

> **Sinh ra bởi:** Không sinh ra - mã nguồn.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `model.py` | `StandardHeadModel` và `CustomHeadModel`; ép checkpoint về fp32; save/load đúng định dạng cho từng loại head. |
| `custom_head.py` | Custom head: 3 view (cls / mean / max) -> LayerNorm -> fusion MLP 2304->512 -> Linear 512->3. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 3 file, 33.3 KB |
| `__init__.py` | file | 579 B · 2026-10-04 10:37:24 |
| `custom_head.py` | file | 10.6 KB · 2026-10-04 11:39:41 |
| `explain.md` | file | 1.7 KB · 2026-10-04 12:58:19 |
| `model.py` | file | 12.3 KB · 2026-10-04 11:00:28 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
