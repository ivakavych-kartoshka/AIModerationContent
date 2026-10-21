# Rule engine voi lexicon Naive Bayes + multi-label

> Validation n=2672, tune threshold rieng tung lexicon (plan §44). NB alpha, nguong multi-label P>=0.25. Khong cham test.

## Ket qua

| Lexicon | Entries | Macro-F1 (thr) | Macro-F1 (argmax) | Accuracy | HATE F1 | OFF F1 | CLEAN F1 |
|---|---|---|---|---|---|---|---|
| current | 668 | **0.4555** | 0.3438 | 0.7021 | 0.3677 | 0.1686 | 0.8301 |
| nb only | 1308 | **0.4782** | 0.3881 | 0.7552 | 0.4036 | 0.1681 | 0.8628 |
| current + nb | 1310 | **0.4596** | 0.3432 | 0.7537 | 0.3619 | 0.1532 | 0.8638 |
| nb multilabel only | 585 | **0.4498** | 0.4191 | 0.8084 | 0.3270 | 0.1223 | 0.9000 |

## Chenh lech + kiem dinh so voi current

| Lexicon | Δ macro-F1 | CI 95% | p (bootstrap) | McNemar +/− | p | Ket luan |
|---|---|---|---|---|---|---|
| nb only | +0.0227 | [+0.0013, +0.0438] | 0.037 | 234/92 | 0.000 | CO y nghia |
| current + nb | +0.0042 | [-0.0168, +0.0247] | 0.702 | 229/91 | 0.000 | khong (CI chua 0) |
| nb multilabel only | -0.0057 | [-0.0359, +0.0245] | 0.708 | 410/126 | 0.000 | khong (CI chua 0) |

## Cach doc

- `nb multilabel only` chi giu term mang CA HAI nhan - kiem tra rieng xem multi-label co phai nguon cai thien khong.
- CI chua 0 => khong co y nghia. Day la rule engine dung rieng, anh huong len fusion con nho hon.
