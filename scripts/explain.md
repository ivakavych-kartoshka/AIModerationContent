<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# scripts - các entry point

> **Dùng để làm gì:** 5 script chạy theo thứ tự tăng dần: chuẩn bị dữ liệu -> kiểm tra môi trường -> train -> đánh giá -> so sánh -> sinh tài liệu folder. Mỗi script chỉ là lớp vỏ mỏng gọi hàm trong `src/`.

> **Sinh ra bởi:** Chỉnh tay.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `00_prepare_data.py` | Tải + làm sạch + tách split ViHSD, kiểm tra SHA-256. Chạy 1 lần đầu tiên. |
| `01_train_all.py` | Train lần lượt cả 4 model. Mỗi model một `run_XXX` riêng. |
| `02_evaluate_all.py` | Chấm test cho các run đã train, tune threshold trên validation, đo tốc độ. |
| `03_compare_models.py` | Gộp kết quả 4 model thành bảng CSV/Markdown, biểu đồ, và file LaTeX cho paper. |
| `04_check_setup.py` | Kiểm tra môi trường, split, config, và forward pass. **Không** train. |
| `05_document_folders.py` | Sinh lại `explain.md` cho toàn bộ cây thư mục + bảng liệt kê các run. |
| `12_build_lexicon.py` | Tải 2 từ điển từ ngữ nhạy cảm từ GitHub về `data/raw/lexicon/` rồi trích xuất danh sách từ đã chuẩn hoá vào `data/processed/lexicon/`. |
| `13_extract_vihsd_words.py` | Quét ViHSD, trích từ/cụm từ xuất hiện nhiều ở HATE+OFFENSIVE và gần như không ở CLEAN, gộp vào lexicon. |
| `14_fetch_additional_datasets.py` | Tải 5 dataset tiếng Việt cùng nhãn CLEAN/OFFENSIVE/HATE về `data/raw/additional/` + file giải thích nhãn `NOTES.md` (ViCM tải từ GitHub, VLSP2019 ký form rồi tự chép file vào). |
| `15_extract_corpus_words.py` | Quét cả 3 dataset (ViHSD/ViCM/bontori), trích ứng viên từ/cụm từ nhạy cảm ra `data/processed/lexicon/sensitive_words_from_datasets.txt`. |
| `16_label_lexicon.py` | Gán nhãn HATE/OFFENSIVE cho từng ứng viên dựa bằng chứng trên **train split**, ghi `verified_lexicon.csv`, `review_queue.csv`, `label_mapping.yaml`, `provenance.json` (plan §36). |
| `17_evaluate_rule_engine.py` | Chấm rule engine trên validation (tune threshold) / test (threshold đóng băng), xuất metrics, confusion matrix, phân tích FP/FN, latency, robustness (plan §39, §44). |
| `18_smoke_test_rule_engine.py` | Smoke test nhanh rule engine (không cần GPU): nạp config/lexicon, kiểm tra offset map + giữ dấu, khớp exact, phân loại biến đổi, xử lý text rỗng/None/rất dài. `--split` (validation/test/train/all) và `--limit 0` chạy end-to-end trên một phần hoặc toàn bộ ViHSD (argmax, không tune threshold). |
| `README.md` | Hướng dẫn dùng từng script. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 3 file, 51.6 KB |
| `00_prepare_data.py` | file | 520 B · 2026-10-04 10:46:09 |
| `01_train_all.py` | file | 6.6 KB · 2026-10-05 07:54:23 |
| `02_evaluate_all.py` | file | 2.6 KB · 2026-10-04 11:25:20 |
| `03_compare_models.py` | file | 673 B · 2026-10-04 11:39:41 |
| `04_check_setup.py` | file | 4.5 KB · 2026-10-04 11:39:41 |
| `05_document_folders.py` | file | 3.2 KB · 2026-10-04 12:36:53 |
| `06_analyze_significance.py` | file | 14.0 KB · 2026-10-15 21:55:28 |
| `07_paper_figures.py` | file | 4.2 KB · 2026-10-15 01:27:03 |
| `08_check_leakage.py` | file | 7.2 KB · 2026-10-15 18:47:38 |
| `09_seed_variance.py` | file | 11.7 KB · 2026-10-15 15:36:49 |
| `10_train_monolingual_baselines.py` | file | 5.6 KB · 2026-10-15 15:33:10 |
| `11_supplementary_analyses.py` | file | 50.5 KB · 2026-10-15 21:19:27 |
| `12_build_lexicon.py` | file | 785 B · 2026-10-16 16:00:36 |
| `13_extract_vihsd_words.py` | file | 788 B · 2026-10-16 17:08:13 |
| `14_fetch_additional_datasets.py` | file | 1.3 KB · 2026-10-16 18:36:46 |
| `15_extract_corpus_words.py` | file | 707 B · 2026-10-16 19:49:20 |
| `16_label_lexicon.py` | file | 816 B · 2026-10-16 23:22:50 |
| `17_evaluate_rule_engine.py` | file | 14.6 KB · 2026-10-16 23:51:56 |
| `18_smoke_test_rule_engine.py` | file | 14.2 KB · 2026-10-17 00:19:24 |
| `explain.md` | file | 4.9 KB · 2026-10-17 00:09:27 |
| `README.md` | file | 7.5 KB · 2026-10-16 18:36:55 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-17 00:20:20. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
