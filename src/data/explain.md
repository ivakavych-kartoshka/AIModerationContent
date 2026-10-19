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
| `corpus_lexicon.py` | Trích ứng viên từ vựng từ các split train (ViHSD/ViCM/bontori), đếm document frequency toxic/clean, ghi `sources/sensitive_words_from_datasets.txt` đầu vào cho bộ gán nhãn. |
| `lexical_labeler.py` | Gán nhãn HATE/OFFENSIVE cho từng ứng viên dựa bằng chứng train-only (share/lift/confidence), áp cổng lọc nhiễu TF-IDF, ghi `verified_lexicon.csv`, `label_mapping.yaml`, `provenance.json` (gốc) + `labeling/candidates_train.csv`, `labeling/review_queue.csv` (plan §36). |
| `tfidf_filter.py` | Bộ lọc nhiễu TF-IDF thuần (không I/O) cho ứng viên lexicon: IDF làm mượt, trọng số TF-IDF theo lớp, tỉ lệ toxic và điểm tách toxic-vs-clean; quyết định giữ/loại (plan §36). |
| `lexicon_pruner.py` | Chấm lại **từ điển đã đóng băng** bằng cổng TF-IDF trên train split, ghi `pruning/verified_lexicon_pruned.csv` (gộp) + bản theo từng dataset `pruning/verified_lexicon_pruned_<dataset>.csv` + audit `pruning/lexicon_prune_audit.csv` và cập nhật `lexicon_report.json`; CLI cho `scripts/20_prune_lexicon.py` (plan §36). |
| `lexicon_relabeler.py` | **Phân nhãn lại** tập đã lọc: chạy lại thuật toán gán nhãn (share/lift/confidence, bằng chứng gộp train split) trên `pruning/verified_lexicon_pruned.csv`, ghi `relabeling/verified_lexicon_relabeled.csv` (gộp) + bản theo từng dataset `relabeling/verified_lexicon_relabeled_<dataset>.csv` + audit trước/sau `relabeling/lexicon_relabel_changes.csv`, cập nhật mục `relabeling` của `lexicon_report.json`; CLI cho `scripts/22_relabel_lexicon.py` (plan §36). |
| `prepare_data.py` | CLI cho `scripts/00_prepare_data.py`. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 14 file, 281.1 KB |
| `__init__.py` | file | 850 B · 2026-10-04 11:21:12 |
| `additional.py` | file | 29.8 KB · 2026-10-17 01:00:08 |
| `corpus_lexicon.py` | file | 19.8 KB · 2026-10-19 12:25:11 |
| `dataset.py` | file | 8.4 KB · 2026-10-04 11:21:49 |
| `download.py` | file | 2.9 KB · 2026-10-04 10:35:16 |
| `explain.md` | file | 7.7 KB · 2026-10-19 15:06:20 |
| `lexical_labeler.py` | file | 40.4 KB · 2026-10-19 12:26:46 |
| `lexicon.py` | file | 15.2 KB · 2026-10-19 12:24:19 |
| `lexicon_pruner.py` | file | 15.9 KB · 2026-10-19 12:27:24 |
| `lexicon_relabeler.py` | file | 14.5 KB · 2026-10-19 12:41:12 |
| `prepare_data.py` | file | 6.0 KB · 2026-10-04 11:40:09 |
| `preprocess.py` | file | 5.9 KB · 2026-10-04 10:35:29 |
| `split.py` | file | 12.8 KB · 2026-10-04 10:35:57 |
| `tfidf_filter.py` | file | 8.1 KB · 2026-10-17 12:49:50 |
| `vihsd_lexicon.py` | file | 12.6 KB · 2026-10-19 12:25:37 |

## Chi tiết

### Cách tính `confidence` - độ chắc chắn của một entry từ điển

Cột `confidence` trong `verified_lexicon.csv` trả lời câu hỏi *"term này chắc chắn là từ độc hại đến đâu"* - **không** phải tần suất xuất hiện thô. Nó do `lexical_labeler._confidence()` tính từ bằng chứng **chỉ trên train split** (plan §36.1).

**Bước 1 - đếm bằng chứng (train-only).** Quét từng câu train của 3 bộ (ViHSD / ViCM / bontori), đếm document frequency theo nhãn (mỗi câu chỉ bầu một lần):

- `n_hate`, `n_offensive`, `n_clean`: số câu mỗi lớp chứa term;
- `r_hate = n_hate / (số câu HATE)`, tương tự `r_offensive`, `r_clean` - **chuẩn hoá theo kích thước lớp** nên lớp CLEAN đông không lấn át;
- `share_hate = r_hate / (r_hate + r_offensive)`: term nghiêng về lớp độc hại nào;
- `lift_hate = r_hate / r_clean` (và `lift_offensive`): term xuất hiện ở HATE nhiều hơn CLEAN bao nhiêu lần.

**Bước 2 - công thức.** `confidence` là trung bình có trọng số của **3 tín hiệu**, mỗi tín hiệu đã được chuẩn hoá về `[0, 1]`:

```
value      = 0.4 * ev + 0.3 * margin + 0.3 * lift_score
confidence = clip(value, 0.05, 0.99)
```

| Thành phần | Công thức | Ý nghĩa | Khoảng |
|---|---|---|---|
| `ev` (evidence) | `min(1, total_toxic / 10)` với `total_toxic = n_hate + n_offensive` | càng nhiều câu độc hại chứa term càng chắc; đủ 10 câu là bão hoà | 0-1 |
| `margin` | `abs(share_hate - 0.5) * 2` | nghiêng hẳn về một lớp -> chắc; `share = 0.5` (lưỡng lự) -> 0 | 0-1 |
| `lift_score` | `log1p(min(lift, 100)) / log1p(100)` | lift càng lớn càng chắc nhưng nén bằng log, trần 100 | 0-1 |

Trọng số `0.4 / 0.3 / 0.3` (`confidence_weights`) và các mốc `10` / `100` đều đổi được trong `LabelingConfig`.

**Ví dụ.** Term có `n_hate = 12`, `n_offensive = 2`, `n_clean = 1` (`total_toxic = 14`), `share_hate ~ 0.9`, `lift ~ 40`:

- `ev = min(1, 14/10) = 1.000` -> `0.4 * 1.000 = 0.400`
- `margin = |0.9 - 0.5| * 2 = 0.800` -> `0.3 * 0.800 = 0.240`
- `lift_score = log1p(40) / log1p(100) = 0.805` -> `0.3 * 0.805 = 0.242`
- `value = 0.400 + 0.240 + 0.242 = 0.881` -> **`confidence = 0.881`**

**Bước 3 - dùng ở đâu.**

- Vào `match_score` của rule engine: `severity x confidence x match_factor x context_factor` (plan §39);
- Xếp loại entry (`review_status`): `confidence >= 0.7` **và** `total_toxic >= 10` -> `verified`; `>= 0.4` -> `auto`; còn lại -> `needs_review`.

Lưu ý: `severity` **không** tính từ dữ liệu - đây là bảng cố định theo nhãn (`HATE = 0.90`, `OFFENSIVE = 0.60`); chỉ `confidence` mới suy ra từ bằng chứng train.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:39. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
