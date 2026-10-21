# Kiem dinh y nghia: lexicon mo rong bang tokenizer tieng Viet

> Tren **validation** (n=2672), 10000 lan paired bootstrap (seed=42). Tune threshold rieng cho tung lexicon (plan §44). Khong cham test.

## Diem (validation)

| Lexicon | Entries | Macro-F1 |
|---|---|---|
| current | 668 | 0.4555 |
| current_plus_word | 929 | 0.4628 |
| word_only | 261 | 0.4465 |

## Kiem dinh so voi lexicon hien tai

| So sanh | Δ macro-F1 | CI 95% | p (bootstrap) | McNemar a>b / a<b | p (McNemar) | Ket luan |
|---|---|---|---|---|---|---|
| current_plus_word vs current | +0.0074 | [-0.0045, +0.0192] | 0.217 | 24 / 36 | 0.155 | KHONG co y nghia (CI chua 0) |
| word_only vs current | -0.0089 | [-0.0409, +0.0228] | 0.586 | 463 / 154 | 0.000 | KHONG co y nghia (CI chua 0) |

## Cach doc (trung thuc)

- CI 95% **chua 0** => khong the ket luan lexicon mo rong tot hon; chenh lech nam trong nhieu.
- Day la so cua **rule engine dung rieng**. Trong he thong fusion, rule chi chiem trong so nho, nen anh huong len macro-F1 cuoi cung con nho hon nua.
- McNemar dem so mau doi ket qua dung/sai: cot `a>b` = lexicon moi dung ma cu sai, `a<b` = nguoc lai.
