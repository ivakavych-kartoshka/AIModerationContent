# Tune nguong multi-label + smoothing cua lexicon NB

> Quet tren **validation** n=2672 (plan §44). Moc current = 0.4555.

## Bang quet (sap theo macro-F1)

| alpha | min_prob | Entries | Multi-label | Macro-F1 | Δ vs current |
|---|---|---|---|---|---|
| 0.3 | 0.35 | 1155 | 208 | **0.4800** | +0.0245 |
| 3.0 | 0.35 | 1140 | 182 | **0.4799** | +0.0245 |
| 1.0 | 0.35 | 1149 | 197 | **0.4789** | +0.0234 |
| 1.0 | 0.25 | 1308 | 585 | **0.4782** | +0.0227 |
| 0.3 | 0.15 | 1408 | 1006 | **0.4761** | +0.0206 |
| 3.0 | 0.25 | 1319 | 688 | **0.4749** | +0.0194 |
| 0.3 | 0.25 | 1318 | 557 | **0.4740** | +0.0185 |
| 1.0 | 0.15 | 1407 | 1067 | **0.4736** | +0.0181 |
| 3.0 | 0.15 | 1416 | 1253 | **0.4705** | +0.0151 |
| 1.0 | 0.50 | 808 | 0 | **0.4064** | -0.0490 |
| 0.3 | 0.50 | 842 | 0 | **0.4052** | -0.0502 |
| 3.0 | 0.50 | 674 | 0 | **0.4049** | -0.0506 |

## Diem tot nhat: alpha=0.3, min_prob=0.35

- Macro-F1 = **0.4800** (Δ = +0.0245)
- Kiem dinh vs current: Δ = +0.0245, CI 95% = [+0.0012, +0.0476], p = 0.038
- McNemar: 235 dung / 78 sai (p = 0.000)
- Ket luan: CO y nghia

## Cach doc

- `min_prob` cao -> it entry hon, it nhan 2 hon; thap -> nhieu entry hon.
- Neu macro-F1 it doi qua cac diem => NB da bao hoa, sieu tham so khong phai nguon cai thien chinh.
- Lexicon tot nhat luu o `lexicon_nb_best.csv`.
