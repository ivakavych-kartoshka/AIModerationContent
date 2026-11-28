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
| `15_extract_corpus_words.py` | Quét cả 3 dataset (ViHSD/ViCM/bontori), trích ứng viên từ/cụm từ nhạy cảm ra `data/processed/lexicon/sources/sensitive_words_from_datasets.txt`. |
| `16_label_lexicon.py` | Gán nhãn HATE/OFFENSIVE cho từng ứng viên dựa bằng chứng trên **train split** và áp cổng lọc nhiễu TF-IDF, ghi `verified_lexicon.csv`, `label_mapping.yaml`, `provenance.json` (gốc) + `labeling/candidates_train.csv`, `labeling/review_queue.csv` (plan §36). |
| `17_evaluate_rule_engine.py` | Chấm rule engine trên validation (tune threshold) / test (threshold đóng băng), xuất metrics, confusion matrix, phân tích FP/FN, latency, robustness (plan §39, §44). |
| `18_smoke_test_rule_engine.py` | Smoke test nhanh rule engine (không cần GPU): nạp config/lexicon, kiểm tra offset map + giữ dấu, khớp exact, phân loại biến đổi, xử lý text rỗng/None/rất dài. `--config` (mặc định `configs/rule_engine.yaml`) để chọn từ điển, `--split` (validation/test/train/all) và `--limit 0` chạy end-to-end trên một phần hoặc toàn bộ ViHSD (argmax, không tune threshold); ghi kết quả vào `outputs_smoke/evaluation_smoke/rule_engine/`. |
| `19_plot_rule_engine.py` | Trực quan hoá kết quả rule engine (plan §39, §44): đọc artifact của script 17, chấm lại bằng threshold đã đóng băng (không tune) và vẽ confusion matrix (thô + chuẩn hoá), per-class F1, PR/ROC, đường tune threshold, robustness (thêm baseline `clean` + mức giảm), histogram latency; thêm `rule_scores.csv` per-sample cho bước fusion. Ghi PNG vào `outputs/evaluation/rule_engine/` và `.../test/`. |
| `20_prune_lexicon.py` | Chấm lại **toàn bộ** từ điển đã đóng băng bằng cổng TF-IDF trên train split và ghi bản đã lọc `pruning/verified_lexicon_pruned.csv` + audit từng entry `pruning/lexicon_prune_audit.csv`; tách thêm bản theo từng dataset (`pruning/verified_lexicon_pruned_<dataset>.csv`, hợp của chúng = file gộp); `--in-place` ghi đè `verified_lexicon.csv` (backup ở `pruning/verified_lexicon_preprune.csv`). Dùng để loại nốt nhiễu lọt qua lúc gán nhãn (plan §36). |
| `21_sweep_tfidf.py` | Quét lưới ngưỡng TF-IDF (`min_ratio`, `min_score`) trên **validation**: mỗi điểm dựng lexicon đã lọc rồi chấm rule engine và ghi macro-F1 vào `outputs/evaluation/rule_engine_tfidf_sweep/`; không bao giờ dùng test để tune (plan §36, §44). |
| `22_relabel_lexicon.py` | **Phân nhãn lại** tập đã lọc: đọc `pruning/verified_lexicon_pruned.csv`, chạy lại thuật toán gán nhãn (share/lift/confidence, bằng chứng gộp mọi train split) trên đúng các term đó, ghi từ điển mới `relabeling/verified_lexicon_relabeled.csv` + bản theo từng dataset `relabeling/verified_lexicon_relabeled_<dataset>.csv` + audit trước/sau `relabeling/lexicon_relabel_changes.csv`; cập nhật mục `relabeling` trong `lexicon_report.json` (plan §36). |
| `26_rule_significance.py` | **Kiểm định chênh lệch giữa 2 lần chạy rule engine** trên cùng test split: paired bootstrap (macro-F1/accuracy, CI 95%, p hai phía) + McNemar đúng/sai từng mẫu, so hàng theo `index` + `y_true` (không khớp là dừng). Input là `predictions.csv` đã chấm sẵn nên **không** tạo lần đo test mới; ghi JSON qua `--out` (mặc định `outputs/evaluation/rule_significance.json`) + bảng LaTeX `papers/tables/rule_significance.tex` (plan §43-44). |
| `27_tune_rule_weights.py` | **Tune trọng số scoring của rule engine** (`severity`, `match_factors`, `negation/quote/context_required_factor`, `aggregation_cap`) bằng coordinate descent trên **validation**; mỗi ứng viên tune lại threshold (macro-F1), mọi trial ghi vào `outputs/evaluation/rule_engine_tune/trials.csv`, cấu hình thắng ghi `configs/rule_engine_tuned.yaml`. Không bao giờ đọc test (plan §39, §44). |
| `28_compare_rule_engines.py` | **So sánh 2 rule engine đã đóng băng** (baseline vs tuned): đọc `metrics.json`/`robustness.json`/`latency.json`/`thresholds.json` của cả hai run ở validation + test, diff cấu hình từ YAML, ghép báo kiểm định của script 26; ghi `outputs/evaluation/comparison/rule_engine/rule_engine_baseline_vs_tuned.md` + 3 hình (macro-F1, per-class F1, robustness). Không chấm lại split nào (plan §44). |
| `29_plot_fusion_comparison.py` | **Hình so sánh fusion với AI-only**: đọc `metrics.json` từng model + `fusion_significance.json` đã có và sinh 4 hình vào `outputs/evaluation/fusion/` (macro-F1 + Δ, forest plot CI paired bootstrap, Δ của 4 biến thể fusion, F1 từng lớp). Chỉ đọc artifact đóng băng, không chấm lại split nào (plan §44). |
| `30_plot_system_architecture.py` | **Sơ đồ kiến trúc toàn hệ thống** cho bài báo: vẽ nhánh AI (mDeBERTa-v3-base + multi-view head → xác suất) + nhánh rule (normalise → lexicon 668 term → điểm rule) hợp nhất bằng fusion `p' = p + β·r` (AI giữ trọng số 1, rule hạ trọng số β) rồi ngưỡng per-channel đóng băng; ghi `papers/figures/system_architecture.png`. Chỉ vẽ, không đọc dữ liệu thực nghiệm nào. |
| `31_ensemble_experiment.py` | **Thí nghiệm ensemble**: trung bình xác suất của 3 model đa dạng (`ens3`) hoặc cả 6 (`ens6`), có/không cộng `beta·rule_scores`; β và ngưỡng chọn trên **validation**, 4 ứng viên đóng băng rồi chấm test **một lần**; kiểm định paired bootstrap + McNemar của ứng viên chính (chọn theo val) và phần post-hoc cho ens6; ghi `outputs/evaluation/ensemble/` (plan §43-44). |
| `32_plot_ensemble_comparison.py` | **Hình so sánh ensemble**: đọc `ensemble_metrics.json` + `significance.json` + số liệu 2 hệ thống tham chiếu (curriculum fusion, dlr2stage) và sinh 4 hình vào `outputs/evaluation/ensemble/` (bar test macro-F1, val vs test đảo chiều, forest plot CI paired bootstrap, F1 từng lớp). Chỉ đọc artifact đóng băng, không chấm lại split nào (plan §44). |
| `README.md` | Hướng dẫn dùng từng script. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 9 file, 170.6 KB |
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
| `13_extract_vihsd_words.py` | file | 811 B · 2026-10-17 01:00:08 |
| `14_fetch_additional_datasets.py` | file | 1.4 KB · 2026-10-17 01:00:08 |
| `15_extract_corpus_words.py` | file | 729 B · 2026-10-17 01:00:08 |
| `16_label_lexicon.py` | file | 847 B · 2026-10-19 12:38:11 |
| `17_evaluate_rule_engine.py` | file | 15.6 KB · 2026-10-19 10:48:24 |
| `18_smoke_test_rule_engine.py` | file | 14.6 KB · 2026-10-17 14:10:59 |
| `19_plot_rule_engine.py` | file | 13.5 KB · 2026-10-17 10:30:26 |
| `20_prune_lexicon.py` | file | 805 B · 2026-10-17 13:05:20 |
| `21_sweep_tfidf.py` | file | 782 B · 2026-10-17 13:06:41 |
| `22_relabel_lexicon.py` | file | 908 B · 2026-10-19 12:38:11 |
| `23_evaluate_fusion.py` | file | 1.1 KB · 2026-10-19 06:51:23 |
| `24_robustness_comparison.py` | file | 24.1 KB · 2026-10-19 08:08:13 |
| `25_fusion_significance.py` | file | 9.8 KB · 2026-10-19 13:25:46 |
| `26_rule_significance.py` | file | 11.0 KB · 2026-10-19 10:53:39 |
| `27_tune_rule_weights.py` | file | 22.2 KB · 2026-10-19 12:57:57 |
| `28_compare_rule_engines.py` | file | 17.7 KB · 2026-10-19 15:09:04 |
| `29_plot_fusion_comparison.py` | file | 11.2 KB · 2026-10-19 15:04:37 |
| `30_plot_system_architecture.py` | file | 9.0 KB · 2026-10-20 06:39:03 |
| `31_ensemble_experiment.py` | file | 20.7 KB · 2026-10-20 07:10:29 |
| `32_plot_ensemble_comparison.py` | file | 15.2 KB · 2026-10-20 07:57:10 |
| `explain.md` | file | 10.5 KB · 2026-10-20 07:13:19 |
| `README.md` | file | 21.0 KB · 2026-10-20 07:58:28 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-20 07:58:43. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
