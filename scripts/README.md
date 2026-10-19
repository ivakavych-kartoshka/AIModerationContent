# SensitiveAI — pipeline scripts

Chạy theo đúng thứ tự. Không tự train ở bước kiểm tra.

## 0. `00_prepare_data.py` — dữ liệu

```powershell
python scripts/00_prepare_data.py
python scripts/00_prepare_data.py --stage preprocess split     # bỏ qua download
python scripts/00_prepare_data.py --stage summary verify       # chỉ kiểm tra
python scripts/00_prepare_data.py --show-samples 3             # xem vài mẫu đã xử lý
```

Tạo `data/processed/vihsd/*_clean.csv`, `data/splits/{train,validation,test}.csv`
và `data/splits/split_manifest.json` (SHA-256 từng split).

## 1. `01_train_all.py` — train 4 model

```powershell
python scripts/01_train_all.py                              # cả 4 model
python scripts/01_train_all.py --models sensitiveai-vi     # chỉ 1 model
python scripts/01_train_all.py --set training.epochs=8     # override config
python scripts/01_train_all.py --dry-run                    # in kế hoạch, không train
```

Kết quả: `reports/<model>/run_00N/` và `experiments/<model>/run_00N/`.

Model 3 và 4 nối tiếp từ checkpoint của model trước (`encoder_checkpoint` trong
config). Đường dẫn ghi trong config thuộc về một run cụ thể, nên nếu không còn thì
`src/common/paths.py:resolve_encoder_checkpoint` tự lấy **run mới nhất** của model
đó — train từng model riêng lẻ vẫn chạy được, chỉ cần model trước đã có ít nhất
một run hoàn tất.

## 2. `02_evaluate_all.py` — chấm test

```powershell
python scripts/02_evaluate_all.py
python scripts/02_evaluate_all.py --models sensitiveai-vi-custom-curriculum
python scripts/02_evaluate_all.py --no-thresholds --force-argmax
python scripts/02_evaluate_all.py --no-benchmark            # bỏ đo latency
```

Threshold được tune trên validation; test chỉ chấm một lần.

## 3. `03_compare_models.py` — bảng + hình cho paper

```powershell
python scripts/03_compare_models.py
```

Ghi `evaluation/comparison/`, copy hình vào `paper/figures/` và sinh
`paper/tables/*.tex` trực tiếp từ kết quả đo.

## 4. `04_check_setup.py` — kiểm tra môi trường

```powershell
python scripts/04_check_setup.py
```

Kiểm tra CUDA, 4 config, fingerprint của fixed split và forward pass của cả
hai loại head. Không train.

## 5. `09_seed_variance.py` — multi-seed (độ ổn định theo seed)

```powershell
python scripts/09_seed_variance.py --seeds 42 43 44 45 46
python scripts/09_seed_variance.py --seeds 43 --models sensitiveai-vi
python scripts/09_seed_variance.py --seeds 42 43 --dry-run   # chỉ sinh config
```

Sáu hệ thống không độc lập (M3 nối từ M2, M4 từ M3, M5 từ M2 + stage_1 của M3,
M6 từ M1). Script tự sinh config riêng cho **từng seed** và ghi lại mọi đường dẫn
`encoder_checkpoint` / `stages[*].init_from` để mỗi seed là một chuỗi khép kín
(không lẫn checkpoint giữa các seed). Kết quả trong `outputs/seeds/`:
`seed_<S>/seed_summary.json` và `seed_variance_summary.{json,md}` (mean ± std).

## 6. `10_train_monolingual_baselines.py` — baseline đơn ngữ

```powershell
python scripts/10_train_monolingual_baselines.py                 # PhoBERT + ViSoBERT
python scripts/10_train_monolingual_baselines.py --models phobert-base
```

Train PhoBERT (`vinai/phobert-base-v2`) và ViSoBERT (`uitnlp/visobert`) bằng
đúng công thức baseline (standard head, cross-entropy không weight, argmax) trên
cùng split đã cố định. Kết quả trong `outputs/baselines/monolingual_summary.{json,md}`.

## 7. `12_build_lexicon.py` — từ điển từ ngữ nhạy cảm

```powershell
python scripts/12_build_lexicon.py                      # tải + trích xuất + tổng kết
python scripts/12_build_lexicon.py --stage download     # chỉ tải 2 lexicon GitHub
python scripts/12_build_lexicon.py --stage extract summary --show-terms 10
python scripts/12_build_lexicon.py --force-download     # tải lại file gốc
```

Tải 2 lexicon tiếng Việt từ GitHub về `data/raw/lexicon/` (giữ nguyên, bất biến):

- `behitek/vietnam-sensitive-words` — profanity, teencode, từ nhạy cảm chính trị
- `blue-eyes-vn/vietnamese-offensive-words` — offensive words + biến thể không dấu / teencode

Rồi chuẩn hoá (NFC, hạ chữ, `_`→space, gộp space), khử trùng lặp và ghi vào
`data/processed/lexicon/`: `behitek_sensitive_words.txt`,
`blueeyes_offensive_words.txt`, `sensitive_words_combined.txt` (danh sách gộp
dùng để quét văn bản) và `lexicon_report.json` (số từ, phần giao, SHA-256).

## 8. `13_extract_vihsd_words.py` — trích từ nhạy cảm từ ViHSD

```powershell
python scripts/13_extract_vihsd_words.py                       # quét train+dev+test
python scripts/13_extract_vihsd_words.py --min-toxic-df 10     # chặt hơn
python scripts/13_extract_vihsd_words.py --show-terms 20       # xem mẫu từ mới
```

Quét các split đã làm sạch trong `data/processed/vihsd/`, đếm document
frequency theo nhóm toxic (HATE+OFFENSIVE) và CLEAN. Từ/cụm từ xuất hiện
`>= --min-toxic-df` lần ở toxic và `<= --max-clean-df` lần ở clean, chưa có
dưới lexicon, sẽ được ghi vào `data/processed/lexicon/sources/vihsd_extracted_words.txt`
và gộp vào `sources/sensitive_words_combined.txt` (cập nhật `lexicon_report.json`).
Chạy `12_build_lexicon.py` trước.

## 9. `14_fetch_additional_datasets.py` — 5 dataset cùng nhãn ViHSD

```powershell
python scripts/14_fetch_additional_datasets.py                 # tải + kiểm tra + tổng kết
python scripts/14_fetch_additional_datasets.py --stage verify  # chỉ đếm lại nhãn
python scripts/14_fetch_additional_datasets.py --force-download
```

Tải về `data/raw/additional/` (bất biến) 5 dataset đã xác minh dùng **đúng**
bộ nhãn `CLEAN`/`OFFENSIVE`/`HATE`:

| Dataset | Nguồn | Điểm đặc biệt |
|---|---|---|
| `vominhmanh/vihsd-explainable` | Hugging Face | ViHSD + `explanation` + `evidence` spans (nguồn từ nhạy cảm chất lượng cao) |
| `leminhhung0101/R-ViHSD` | Hugging Face | 48K mẫu train, biến thể nhiễu (`noise_type`) — kiểm tra độ robust |
| `BonTori/vietnamese-hsd-cleaned` | Hugging Face | ~62K **mẫu mới** từ TDTU YouTube/voz, cùng 3 nhãn |
| `ViCM` | GitHub | 5.415 **mẫu mới** code-mixed (+480 diễn đàn); nhãn `0=Clean,1=Hate,2=Offensive` → **phải remap** |
| `VLSP2019-HSD` | thủ công | 25.431 **mẫu mới** bài Facebook; phải ký agreement form |

Bước verify đếm nhãn thật trên từng file, ghi `NOTES.md` mỗi dataset và
`additional_datasets_report.json` tổng hợp (`labels_exact_match`,
`source`, `requires_remap`, `pending_manual_download`).

Riêng **VLSP2019-HSD** không tải tự động được (không có mirror công khai):

1. Mở trang task: https://vlsp.org.vn/vlsp2019/eval/hsd
2. Ký **HSD Agreement form**: https://drive.google.com/drive/folders/1HSOWdiMFQy0bjd09TBYaI7lsf7iYZaq5
3. Chép file `.csv`/`.xlsx` nhận được vào `data/raw/additional/vlsp2019-hsd/`.
4. Chạy lại `python scripts/14_fetch_additional_datasets.py --stage verify`.

## 10. `19_plot_rule_engine.py` — hình cho rule engine

```powershell
python scripts/19_plot_rule_engine.py                 # hình cho validation
python scripts/19_plot_rule_engine.py --split test    # hình cho test (threshold đóng băng)
```

Đọc artifact của `17_evaluate_rule_engine.py`, chấm lại bằng threshold đã đóng băng
(**không tune**) rồi vẽ: confusion matrix (thô + chuẩn hoá), per-class F1, PR/ROC,
đường tune threshold (validation), robustness theo biến đổi (kèm baseline `clean`
và `detection_drop_vs_clean`), histogram latency. Đồng thời ghi `rule_scores.csv`
per-sample (đầu vào cho bước fusion).

Hình đi vào `outputs/evaluation/rule_engine/*.png` và `.../test/*.png`; không đụng
tới `metrics.json` của script 17. Chạy `17_evaluate_rule_engine.py` trước.

## 11. `16_label_lexicon.py` — gán nhãn + lọc nhiễu TF-IDF

```powershell
python scripts/16_label_lexicon.py                                  # gán nhãn + lọc nhiễu TF-IDF (mặc định)
python scripts/16_label_lexicon.py --no-tfidf                       # tắt cổng TF-IDF (chỉ gán nhãn như cũ)
python scripts/16_label_lexicon.py --tfidf-min-score 0.003          # lọc nhẹ tay hơn (~10% bị loại)
python scripts/16_label_lexicon.py --tfidf-min-score 0.005          # lọc mạnh tay hơn (~47% bị loại)
python scripts/16_label_lexicon.py --tfidf-min-ratio 0.7 --tfidf-smoothing 1.0
python scripts/16_label_lexicon.py --validate-only
```

Gán nhãn `HATE`/`OFFENSIVE` cho từng ứng viên dựa bằng chứng **train-only**
(share/lift/confidence), rồi áp **cổng lọc nhiễu TF-IDF** (`src/data/tfidf_filter.py`):
mỗi từ được tính IDF làm mượt + trọng số TF-IDF theo lớp (TF = document frequency,
chuẩn hoá theo cỡ lớp), từ đó ra `tfidf_ratio` (tỉ lệ trọng số nằm ở lớp toxic) và
`tfidf_score` (mức tách toxic-vs-clean có trọng số IDF). Từ bị loại khi
`tfidf_ratio < min_ratio` hoặc `tfidf_score < min_score`; chúng không vào
`verified_lexicon.csv` nên không tới rule engine, và được ghi vào `review_queue.csv`
với `reason` (`tfidf_clean_dominant`, `tfidf_low_score`, `tfidf_no_toxic`).

Các cột TF-IDF (`tf_*`, `idf`, `tfidf_*`) cũng được thêm vào `candidates_train.csv`.
Output: `data/processed/lexicon/` giữ `verified_lexicon.csv`, `label_mapping.yaml`,
`provenance.json` (kèm tham số TF-IDF); hai bảng bằng chứng trung gian
`candidates_train.csv`, `review_queue.csv` ghi vào `data/processed/lexicon/labeling/`.
Ghi ra thư mục khác bằng `--output-dir` nếu muốn so sánh (không ghi đè lexicon chính).

## 12. `20_prune_lexicon.py` — lọc lại toàn bộ từ điển bằng TF-IDF

```powershell
python scripts/20_prune_lexicon.py                                   # audit, dùng ngưỡng mặc định (0.6 / 0.004)
python scripts/20_prune_lexicon.py --tfidf-min-score 0.006           # áp ngưỡng tốt nhất từ script 21
python scripts/20_prune_lexicon.py --tfidf-min-score 0.006 --in-place  # ghi đè verified_lexicon.csv (có backup)
python scripts/20_prune_lexicon.py --dry-run                         # chỉ in quyết định, không ghi file
```

Script 16 đã áp cổng TF-IDF lúc gán nhãn, nhưng từ điển **đã đóng băng** vẫn có thể
lẫn nhiễu: entry do người duyệt tay (`manual_review.csv`) nâng lên, từ nhập từ lexicon
ngoài, hoặc từ được đóng băng bằng ngưỡng cũ. Script 20 chấm lại **mọi entry** của
`verified_lexicon.csv` trên train split và ghi vào `data/processed/lexicon/pruning/`:

- `verified_lexicon_pruned.csv` — bản đã lọc **gộp tất cả dataset** (giữ nguyên schema);
- `verified_lexicon_pruned_vihsd.csv`, `verified_lexicon_pruned_vicm.csv`,
  `verified_lexicon_pruned_bontori.csv` — bản đã lọc **riêng từng dataset** (theo cột
  `source_datasets`); entry thuộc nhiều dataset nằm ở nhiều file, nên **hợp của các file
  riêng = file gộp**. Tắt bằng `--no-per-dataset`.
- `lexicon_prune_audit.csv` — mỗi entry kèm số đếm nhãn, `idf`, `tfidf_*`, quyết định
  `kept`/`dropped` và lý do (`tfidf_low_score`, `tfidf_clean_dominant`, `unseen_in_train`);
- `verified_lexicon_preprune.csv` — backup của `verified_lexicon.csv` (chỉ khi `--in-place`);
- `lexicon_report.json` (ở gốc lexicon) — thêm mục `tfidf_pruning` (kèm `kept_by_dataset`).

## 13. `21_sweep_tfidf.py` — quét ngưỡng TF-IDF trên validation

```powershell
python scripts/21_sweep_tfidf.py
python scripts/21_sweep_tfidf.py --ratios 0.5 0.6 0.7 --scores 0.004 0.005 0.006
python scripts/21_sweep_tfidf.py --limit 500        # chạy nhanh để thử
```

Quét lưới `(min_ratio, min_score)`: mỗi điểm dựng lexicon đã lọc rồi chấm rule engine
trên **validation** (tune threshold per-class, mục tiêu macro-F1) và ghi kết quả vào
`outputs/evaluation/rule_engine_tfidf_sweep/tfidf_sweep.{json,csv}` (kèm `baseline`
không lọc và `best`). **Không bao giờ dùng test để tune** (plan §44) — lấy `min_score`
của `best` rồi áp bằng script 20.

## 14. `22_relabel_lexicon.py` — phân nhãn lại tập đã lọc

```powershell
python scripts/22_relabel_lexicon.py
python scripts/22_relabel_lexicon.py --share-threshold 0.7 --dry-run
python scripts/22_relabel_lexicon.py --no-tfidf
```

Script 20 chỉ **loại bớt** entry; các entry còn lại vẫn giữ nhãn cũ. Script 22 chạy lại
**thuật toán gán nhãn** (share/lift/confidence, bằng chứng gộp mọi train split) trên đúng
tập đã lọc `pruning/verified_lexicon_pruned.csv` và ghi vào
`data/processed/lexicon/relabeling/`:

- `verified_lexicon_relabeled.csv` — từ điển phân nhãn lại **gộp tất cả dataset** (giữ nguyên schema);
- `verified_lexicon_relabeled_vihsd.csv`, `verified_lexicon_relabeled_vicm.csv`,
  `verified_lexicon_relabeled_bontori.csv` — bản riêng **từng dataset** (theo cột
  `source_datasets`); hợp của các file riêng = file gộp. Tắt bằng `--no-per-dataset`.
- `lexicon_relabel_changes.csv` — audit trước/sau mỗi term (nhãn, `review_status`,
  `confidence`, `severity`, cờ `changed`, lý do);
- `lexicon_report.json` (ở gốc lexicon) — thêm mục `relabeling`.

Vì bằng chứng gộp không đổi, chạy với tham số mặc định cho **cùng nhãn** như script 20
(`changed=0`); đổi `--share-threshold`/`--clean-lift`/`--no-tfidf` để xem nhãn thay đổi.

## 15. `27_tune_rule_weights.py` — tune trọng số scoring trên validation

```powershell
python scripts/27_tune_rule_weights.py                     # coordinate descent, 3 vòng
python scripts/27_tune_rule_weights.py --rounds 1 --limit 800   # thử nhanh
python scripts/27_tune_rule_weights.py --no-write-config   # chỉ search, không ghi YAML
```

`scripts/17` chỉ tune **threshold**; script 27 tune chính **trọng số** của plan §39:
`scoring.severity` (HATE/OFFENSIVE), `scoring.match_factors` (7 kiểu khớp),
`negation_factor`, `quote_factor`, `context_required_factor`, `aggregation_cap`.
Thuật toán là coordinate descent tham lam: mỗi vòng thử từng giá trị trong lưới của
từng tham số, giữ lại giá trị làm **macro-F1 validation** tăng quá `--min-gain`
(mặc định 0.001); mỗi ứng viên đều **tune lại threshold trên validation** nên trọng số
và ngưỡng luôn đi đôi với nhau. Dừng sớm khi một vòng không còn cải thiện.

Giao thức plan §44: script chỉ đọc `data/splits/validation.csv`, **không bao giờ** load
test; mọi trial ghi vào `outputs/evaluation/rule_engine_tune/trials.csv` +
`tuning_report.json`; cấu hình thắng ghi `configs/rule_engine_tuned.yaml` (kèm header
ghi baseline -> best, các tham số đã đổi, SHA-256 từ điển). Sau khi đóng băng trọng số,
chạy `17 --split validation` để freeze `thresholds.json` rồi mới chấm test **một lần**.

## 16. `26_rule_significance.py` — kiểm định chênh lệch 2 lần chạy rule engine

```powershell
python scripts/26_rule_significance.py --a outputs/evaluation/rule_engine_tuned/test/predictions.csv `
        --b outputs/evaluation/rule_engine/test/predictions.csv --label-a rule_engine_tuned --label-b rule_engine
```

Đọc hai `predictions.csv` **đã chấm sẵn** (không chạy lại rule engine, không tạo thêm lần
đo test - plan §44), so hàng theo `index` + `y_true` (sai thứ tự là dừng ngay) rồi báo:
CI bootstrap 95% cho từng hệ thống, paired bootstrap cho Δ (macro-F1, accuracy) kèm p hai
phía, và McNemar exact trên vector đúng/sai từng mẫu. JSON ghi theo `--out` (mặc định
`outputs/evaluation/rule_significance.json`), bảng LaTeX theo `--tex` (`--no-tex` để bỏ),
`--n-boot`/`--seed`/`--alpha` chỉnh bootstrap.

Kết quả cặp baseline/tuned hiện tại: Δ macro-F1 **+0.0799** CI [0.0598, 0.1002] p < 1e-4,
McNemar p = 2.2e-136 → có ý nghĩa thống kê.

## 17. `28_compare_rule_engines.py` — báo so sánh 2 rule engine

```powershell
python scripts/28_compare_rule_engines.py                        # baseline vs tuned, cả 2 split
python scripts/28_compare_rule_engines.py --significance <json>  # dùng số kiểm định khác
```

Chỉ **đọc** artifact của `scripts/17`/`19` (`metrics.json`, `robustness.json`,
`latency.json`, `thresholds.json`) + diff cấu hình từ `configs/rule_engine{,_tuned}.yaml`
+ số kiểm định của script 26; không chấm lại bất kỳ split nào. Ghi vào
`outputs/evaluation/comparison/rule_engine/`: báo `rule_engine_baseline_vs_tuned.md` (8 mục: khác biệt
cấu hình, validation, test, per-class, robustness, độ trễ, kiểm định, kết luận) và 3 hình
`rule_engine_{macro_f1,per_class_f1,robustness}.png`.

## 18. `29_plot_fusion_comparison.py` — hình so sánh fusion với AI-only

```powershell
python scripts/29_plot_fusion_comparison.py                      # 6 model, 4 hình
python scripts/29_plot_fusion_comparison.py --fusion-dir <folder>  # folder khác
```

Chỉ **đọc** `metrics.json` từng model + `fusion_significance.json` (không chấm lại test),
sinh 4 hình vào `outputs/evaluation/fusion/`:

- `fusion_ai_vs_fusion.png` — thanh so macro-F1 test (AI vs fusion primary) + panel Δ zoom;
- `fusion_delta_ci.png` — forest plot Δ = fusion − AI với CI 95% (ngôi sao = significant);
- `fusion_variants_delta.png` — Δ của 4 biến thể (argmax/thresholds/per-channel/meta) so AI;
- `fusion_per_class_f1.png` — F1 HATE/OFFENSIVE/CLEAN của AI vs fusion.

Ký hiệu Δ trên hình là **fusion − AI** (dương = fusion hơn); số trong
`significance_report.md` là chiều ngược lại (AI − fusion) — cùng một kiểm định.

## Module-level (chạy từng bước)

```powershell
python -m src.data.prepare_data          --stage preprocess split
python -m src.training.run_experiment    --config configs/sensitiveai-vi-customdlr2stage.yaml
python -m src.evaluation.run_evaluation  --model sensitiveai-vi --run-id run_001
python -m src.comparison.compare_models  --reports-dir reports
```
