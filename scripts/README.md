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

## Module-level (chạy từng bước)

```powershell
python -m src.data.prepare_data          --stage preprocess split
python -m src.training.run_experiment    --config configs/sensitiveai-vi-customdlr2stage.yaml
python -m src.evaluation.run_evaluation  --model sensitiveai-vi --run-id run_001
python -m src.comparison.compare_models  --reports-dir reports
```
