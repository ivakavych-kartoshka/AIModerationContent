<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# data - dữ liệu

> **Dùng để làm gì:** Tải ViHSD, tiền xử lý, tách split có fingerprint SHA-256, và tạo dataset đã tokenize cho PyTorch.

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Kết quả của nó nằm ở `data/` (thư mục dữ liệu).

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `download.py` | Tải `uitnlp/vihsd` từ Hugging Face về `data/raw/vihsd/`. |
| `preprocess.py` | Làm sạch + remap nhãn canonical + ghi `preprocessing_report.json`. |
| `split.py` | Tạo split, tính SHA-256 từng file, kiểm tra toàn vẹn, ghi `split_manifest.json`. |
| `dataset.py` | Tokenize `train/validation/test`, `TokenizedTextDataset`, `collate_fn` an toàn Windows (num_workers>0). |
| `prepare_data.py` | CLI cho `scripts/00_prepare_data.py`. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 6 file, 59.5 KB |
| `__init__.py` | file | 850 B · 2026-10-04 11:21:12 |
| `dataset.py` | file | 8.4 KB · 2026-10-04 11:21:49 |
| `download.py` | file | 2.9 KB · 2026-10-04 10:35:16 |
| `explain.md` | file | 2.0 KB · 2026-10-04 12:58:19 |
| `prepare_data.py` | file | 6.0 KB · 2026-10-04 11:40:09 |
| `preprocess.py` | file | 5.9 KB · 2026-10-04 10:35:29 |
| `split.py` | file | 12.8 KB · 2026-10-04 10:35:57 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
