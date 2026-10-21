# Rule engine với lexicon mở rộng bằng tokenizer tiếng Việt

> Đo trên **validation** (n=2,672), tune threshold riêng cho từng lexicon (plan §44). Không chấm test. Lexicon chính thức không bị sửa.

## Kết quả

| Lexicon | Số entry | Macro-F1 (threshold) | Macro-F1 (argmax) | Accuracy | HATE F1 | OFF F1 | CLEAN F1 |
|---|---|---|---|---|---|---|---|
| current (verified_lexicon.csv) | 668 | **0.4555** | 0.3438 | 0.7021 | 0.3677 | 0.1686 | 0.8301 |
| current + word terms | 929 | **0.4628** | 0.3436 | 0.6976 | 0.3830 | 0.1783 | 0.8272 |
| word-mode terms only | 261 | **0.4465** | 0.4145 | 0.8177 | 0.3431 | 0.0934 | 0.9031 |

## Chênh lệch so với lexicon hiện tại

| Lexicon | Δ macro-F1 | Δ HATE F1 | Δ OFF F1 |
|---|---|---|---|
| current (verified_lexicon.csv) | +0.0000 | +0.0000 | +0.0000 |
| current + word terms | +0.0074 | +0.0153 | +0.0096 |
| word-mode terms only | -0.0089 | -0.0246 | -0.0752 |

## Cách đọc

- Δ dương = lexicon mở rộng tốt hơn trên validation.
- Nếu Δ ≈ 0 hoặc âm: tokenizer tiếng Việt **không** giúp rule engine — term mới (138) không bù được term mất (270), hoặc term mới không match được text nhiễu.
- Số này là **rule engine đứng riêng**; ảnh hưởng lên fusion còn nhỏ hơn nữa (fusion chỉ chiếm trọng số β nhỏ).
