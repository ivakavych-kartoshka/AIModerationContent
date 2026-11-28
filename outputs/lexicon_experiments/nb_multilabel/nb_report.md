# Mo hinh xac suat (Naive Bayes) + multi-label cho lexicon

> Corpus: ViHSD+ViCM+Bontori (135238 dong). Laplace alpha=1.0, nguong multi-label P>=0.25. Khong cham test.

## Phan bo top-1 label (truoc khi multi-label hoa)

| Top-1 label | So term |
|---|---|
| HATE | 804 |
| OFFENSIVE | 295 |
| CLEAN | 329 |

## Sau multi-label hoa

- Tong entry giu lai: **1308** (prob HATE hoac OFF >= 0.25)
- Trong do **585** entry mang CA HAI nhan `HATE;OFFENSIVE` (matcher se cong diem cho ca 2 class)

## Vi du term xac suat cao

| term | P(HATE) | P(OFF) | P(CLEAN) | nhan giu |
|---|---|---|---|---|
| `thằng mặt` | 0.907 | 0.062 | 0.030 | HATE |
| `thằng chó` | 0.902 | 0.082 | 0.016 | HATE |
| `con mẹ mày` | 0.898 | 0.069 | 0.033 | HATE |
| `mày ngu` | 0.895 | 0.064 | 0.041 | HATE |
| `thằng lồn` | 0.893 | 0.081 | 0.026 | HATE |
| `chó ad` | 0.891 | 0.055 | 0.054 | HATE |
| `nước hại` | 0.891 | 0.055 | 0.054 | HATE |
| `mày là thằng` | 0.891 | 0.055 | 0.054 | HATE |
| `máy tui` | 0.040 | 0.882 | 0.078 | OFFENSIVE |
| `bủm chúa` | 0.063 | 0.876 | 0.061 | OFFENSIVE |
| `cắp content` | 0.063 | 0.876 | 0.061 | OFFENSIVE |
| `ăn cắp content` | 0.063 | 0.876 | 0.061 | OFFENSIVE |
| `t nể` | 0.067 | 0.868 | 0.065 | OFFENSIVE |
| `skinship` | 0.072 | 0.859 | 0.070 | OFFENSIVE |
| `đợi đợi` | 0.072 | 0.859 | 0.070 | OFFENSIVE |
| `thù năm` | 0.096 | 0.858 | 0.046 | OFFENSIVE |

## Cach doc

- `prob_*` la posterior SOFTMAX cua lo-odds Naive Bayes (da bu tru class lech).
- Term mang 2 nhan = term xuat hien nhieu o CA HAI class (da nghia that, vd `dm`, `vl`) - day chinh la thu argmax truoc day lam mat.
- Buoc do F1 tren validation: script 41.
