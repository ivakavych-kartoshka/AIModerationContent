# Paired significance: AI-only (frozen thresholds) vs hybrid fusion

Models: sensitiveai-vi, sensitiveai-vi-custom, sensitiveai-vi-custom-curriculum, sensitiveai-vi-customdlr2stage
Test samples: 6680

## sensitiveai-vi

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): 0.0051 [-0.0020, 0.0120], p=0.159200
- McNemar (correctness): b=61 (A right, B wrong), c=30 (A wrong, B right), n=91, exact p=0.001517

## sensitiveai-vi-custom

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): -0.0027 [-0.0079, 0.0027], p=0.294200
- McNemar (correctness): b=15 (A right, B wrong), c=23 (A wrong, B right), n=38, exact p=0.255875

## sensitiveai-vi-custom-curriculum

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): -0.0028 [-0.0058, 0.0004], p=0.086800
- McNemar (correctness): b=8 (A right, B wrong), c=31 (A wrong, B right), n=39, exact p=0.000294

## sensitiveai-vi-customdlr2stage

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): 0.0000 [-0.0020, 0.0024], p=0.978200
- McNemar (correctness): b=2 (A right, B wrong), c=3 (A wrong, B right), n=5, exact p=1.000000
