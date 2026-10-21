# Tổng kết các hướng cải thiện đã thử

> Tất cả Δ đo trên **rule engine đứng riêng**, validation n=2672 (trừ dòng fusion = test). 10000 paired bootstrap.

| Hướng | Rule engine (val) | Δ | CI 95% | p | Kết luận |
|---|---|---|---|---|---|
| Lexicon hiện tại | 0.4555 | — | — | — | mốc |
| Hướng 1: tokenize VI +word terms | 0.4628 | +0.0074 | [-0.0045, +0.0192] | 0.217 | ✗ không |
| Hướng 2: fastText embedding only | 0.4545 | -0.0010 | [-0.0245, +0.0232] | 0.934 | ✗ không |
| Hướng 3: fastText +embedding | 0.4585 | +0.0030 | [-0.0196, +0.0263] | 0.811 | ✗ không |
| Hướng 4: NB xác suất + multi-label | 0.4782 | +0.0227 | [+0.0013, +0.0438] | 0.037 | ✅ CÓ ý nghĩa |
| Hướng 4 (tuned) alpha=0.3, p=0.35 | 0.4800 | +0.0245 | [+0.0012, +0.0476] | 0.038 | ✅ CÓ ý nghĩa |

## Đo trên FUSION (test)

| Hệ thống | Macro-F1 | Δ | CI 95% | p | Kết luận |
|---|---|---|---|---|---|
| fusion SOTA (668) | 0.6647 | — | — | — | mốc |
| fusion + NB (1155) | 0.6677 | +0.0030 | [-0.0035, +0.0094] | 0.351 | ✗ không |

**Kết luận:** 3/4 hướng thất bại ở mức rule engine. Hướng 4 (mô hình xác suất NB) THÀNH CÔNG ở mức rule engine (+0.0245, p=0.037) nhưng khi đưa vào fusion thì cải thiện biến mất (+0.0030, không ý nghĩa) vì bottleneck thật nằm ở lớp OFFENSIVE và ở AI, không ở rule engine.

