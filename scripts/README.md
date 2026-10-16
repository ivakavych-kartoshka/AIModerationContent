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

## Module-level (chạy từng bước)

```powershell
python -m src.data.prepare_data          --stage preprocess split
python -m src.training.run_experiment    --config configs/sensitiveai-vi-customdlr2stage.yaml
python -m src.evaluation.run_evaluation  --model sensitiveai-vi --run-id run_001
python -m src.comparison.compare_models  --reports-dir reports
```
