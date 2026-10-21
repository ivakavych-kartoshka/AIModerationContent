# Tom tat so sanh cac bien the lexicon (validation)

> Tat ca so do tren validation, tune threshold rieng tung lexicon (plan §44). Khong cham test.

| Lexicon | Entries | Macro-F1 | Δ vs current | CI 95% | p | Mac-Nemar +/− | Ket luan |
|---|---|---|---|---|---|---|---|
| current | 668 | 0.4555 | - | - | - | - | moc |
| +word terms | 929 | 0.4628 | +0.0074 | [-0.0045, +0.0192] | 0.217 | 24.0/36.0 | khong (CI chua 0) |
| word-only | 261 | 0.4465 | -0.0089 | [-0.0409, +0.0228] | 0.586 | 463.0/154.0 | khong (CI chua 0) |
| embedding only | 1207 | 0.4545 | -0.0010 | [-0.0245, +0.0232] | 0.934 | 236/232 | khong (CI chua 0) |
| +embedding | 1272 | 0.4585 | +0.0030 | [-0.0196, +0.0263] | 0.811 | 269/214 | khong (CI chua 0) |

**Ket luan:** khong cach mo rong lexicon nao cai thien macro-F1 mot cach co y nghia thong ke. Xem 4 hinh trong `outputs/lexicon_experiments/figures/`.

