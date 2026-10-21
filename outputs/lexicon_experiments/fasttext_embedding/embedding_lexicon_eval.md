# Rule engine voi lexicon do embedding (fastText) gan nhan

> Tren **validation** (n=2672), tune threshold rieng tung lexicon (plan §44). Nhan term do fastText gan (script 37), khong phai nguoi gan. Khong cham test.

## Ket qua

| Lexicon | Entries | Macro-F1 (thr) | Macro-F1 (argmax) | Accuracy | HATE F1 | OFF F1 | CLEAN F1 |
|---|---|---|---|---|---|---|---|
| current | 668 | **0.4555** | 0.3438 | 0.7021 | 0.3677 | 0.1686 | 0.8301 |
| embedding only | 1207 | **0.4545** | 0.3044 | 0.7036 | 0.3672 | 0.1657 | 0.8306 |
| current + embedding | 1272 | **0.4585** | 0.3091 | 0.7227 | 0.3758 | 0.1569 | 0.8428 |

## Chenh lech + kiem dinh so voi current

| Lexicon | Δ macro-F1 | CI 95% | p (bootstrap) | McNemar a>b / a<b | p | Ket luan |
|---|---|---|---|---|---|---|
| embedding only | -0.0010 | [-0.0245, +0.0232] | 0.934 | 236 / 232 | 0.890 | KHONG (CI chua 0) |
| current + embedding | +0.0030 | [-0.0196, +0.0263] | 0.811 | 269 / 214 | 0.014 | KHONG (CI chua 0) |

## Cach doc

- Δ la so voi lexicon hien tai. CI chua 0 => chenh lech nam trong nhieu.
- Day la rule engine **dung rieng**; anh huong len fusion cuoi cung nho hon nua.
- Nhan do embedding gan co the sai (vd tu trung tinh bi gan OFFENSIVE) - xem script 37.
