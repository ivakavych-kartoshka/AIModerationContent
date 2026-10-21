# Fusion voi lexicon NB vs SOTA hien tai (test)

> n=6680 test, ca hai dung cung AI predictions cache; chenh lech chi den tu lexicon rule. 10000 bootstrap.

## Macro-F1 (test)

| He thong | Macro-F1 | HATE F1 | OFF F1 | CLEAN F1 |
|---|---|---|---|---|
| SOTA | **0.6647** | 0.5920 | 0.4612 | 0.9408 |
| fusion_nb | **0.6677** | 0.6084 | 0.4535 | 0.9412 |

## Kiem dinh (fusion_nb vs SOTA)

- Δ macro-F1 = **+0.0030**
- CI 95% = [-0.0035, +0.0094]
- p (bootstrap) = 0.351
- McNemar: 29 dung / 25 sai (p = 0.6835)
- **Ket luan: KHONG co y nghia (CI chua 0)**

## Cach doc

- Δ duong = lexicon NB day fusion len. Day la so TEST, chi doc lai tu artifact da co.
- AI predictions khong doi, nen chenh lech hoan toan do lexicon rule.
