# Supplementary analyses (frozen predictions)

- test items: 6,680
- bootstrap resamples: 10,000 (seed 42)

## Calibration

| System | ECE | Brier | NLL | mean conf. | MCC |
|---|---|---|---|---|---|
| M1 | 0.106 | 0.2283 | 0.6218 | 0.979 | 0.5541 |
| M2 | 0.111 | 0.2334 | 0.8382 | 0.987 | 0.5533 |
| M3 | 0.120 | 0.2437 | 1.1914 | 0.996 | 0.5650 |
| M4 | 0.119 | 0.2409 | 1.0443 | 0.993 | 0.5550 |
| M5 | 0.118 | 0.2372 | 1.0472 | 0.996 | 0.5538 |
| M6 | 0.068 | 0.2170 | 0.4817 | 0.931 | 0.5636 |

## Error complementarity

| # systems wrong | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| test comments | 5,410 | 228 | 178 | 93 | 143 | 124 | 504 |

- missed by all six: 504 ({'HATE': 185, 'OFFENSIVE': 180, 'CLEAN': 139})
- solved by all six: 5410 ({'HATE': 270, 'OFFENSIVE': 130, 'CLEAN': 5010})

## Prior-reweighting diagnostic (oracle on test)

| System | argmax M-F1 | u* | best M-F1 | gain |
|---|---|---|---|---|
| M1 | 0.6545 | 0.70 | 0.6564 | +0.0018 |
| M2 | 0.6575 | 0.80 | 0.6585 | +0.0009 |
| M3 | 0.6638 | 1.00 | 0.6638 | +0.0000 |
| M4 | 0.6602 | 2.80 | 0.6616 | +0.0014 |
| M5 | 0.6536 | 3.00 | 0.6549 | +0.0012 |
| M6 | 0.6604 | 0.60 | 0.6656 | +0.0051 |
