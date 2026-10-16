<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# data - dữ liệu

> **Dùng để làm gì:** Tải ViHSD, tiền xử lý, tách split có fingerprint SHA-256, và tạo dataset đã tokenize cho PyTorch.

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Kết quả của nó nằm ở `data/` (thư mục dữ liệu).

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `download.py` | Tải `uitnlp/vihsd` từ Hugging Face về `data/raw/vihsd/`. |
| `lexicon.py` | Tải 2 lexicon GitHub về `data/raw/lexicon/`, chuẩn hoá (NFC, hạ chữ, `_`→space) và ghi danh sách từ nhạy cảm + `lexicon_report.json` (SHA-256) vào `data/processed/lexicon/`. |
| `vihsd_lexicon.py` | Quét ViHSD đã làm sạch, đếm document frequency theo nhóm toxic/clean, trích từ/cụm từ mới vào lexicon (ghép `vihsd_extracted_words.txt` + cập nhật report). |
| `additional.py` | Tải 5 dataset cùng nhãn CLEAN/OFFENSIVE/HATE (3 từ Hugging Face, ViCM từ GitHub, VLSP2019 tải thủ công) về `data/raw/additional/`, đếm nhãn từng file (áp `label_map` nếu mã hoá khác), ghi `NOTES.md` từng dataset + `additional_datasets_report.json` tổng hợp. |
| `preprocess.py` | Làm sạch + remap nhãn canonical + ghi `preprocessing_report.json`. |
| `split.py` | Tạo split, tính SHA-256 từng file, kiểm tra toàn vẹn, ghi `split_manifest.json`. |
| `dataset.py` | Tokenize `train/validation/test`, `TokenizedTextDataset`, `collate_fn` an toàn Windows (num_workers>0). |
| `prepare_data.py` | CLI cho `scripts/00_prepare_data.py`. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 11 file, 217.3 KB |
| `__init__.py` | file | 850 B · 2026-10-04 11:21:12 |
| `additional.py` | file | 29.1 KB · 2026-10-16 18:33:57 |
| `corpus_lexicon.py` | file | 15.9 KB · 2026-10-16 20:16:02 |
| `dataset.py` | file | 8.4 KB · 2026-10-04 11:21:49 |
| `download.py` | file | 2.9 KB · 2026-10-04 10:35:16 |
| `explain.md` | file | 3.0 KB · 2026-10-17 00:09:27 |
| `lexical_labeler.py` | file | 35.0 KB · 2026-10-16 23:28:52 |
| `lexicon.py` | file | 14.7 KB · 2026-10-16 16:00:32 |
| `prepare_data.py` | file | 6.0 KB · 2026-10-04 11:40:09 |
| `preprocess.py` | file | 5.9 KB · 2026-10-04 10:35:29 |
| `split.py` | file | 12.8 KB · 2026-10-04 10:35:57 |
| `vihsd_lexicon.py` | file | 12.2 KB · 2026-10-16 17:11:53 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
