# Paired significance: AI-only (frozen thresholds) vs hybrid fusion

Models: sensitiveai-vi, sensitiveai-vi-custom, sensitiveai-vi-custom-curriculum, sensitiveai-vi-customdlr2stage, sensitiveai-vi-customdlr2stage-ext, sensitiveai-vi-stage2
Test samples: 6680

## sensitiveai-vi

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): 0.0034 [-0.0038, 0.0105], p=0.369400
- McNemar (correctness): b=52 (A right, B wrong), c=41 (A wrong, B right), n=93, exact p=0.299732

## sensitiveai-vi-custom

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): -0.0029 [-0.0080, 0.0025], p=0.287200
- McNemar (correctness): b=13 (A right, B wrong), c=22 (A wrong, B right), n=35, exact p=0.175465

## sensitiveai-vi-custom-curriculum

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): -0.0057 [-0.0113, -0.0004], p=0.033000
- McNemar (correctness): b=14 (A right, B wrong), c=40 (A wrong, B right), n=54, exact p=0.000535

## sensitiveai-vi-customdlr2stage

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): 0.0001 [-0.0022, 0.0025], p=0.980200
- McNemar (correctness): b=3 (A right, B wrong), c=3 (A wrong, B right), n=6, exact p=1.000000

## sensitiveai-vi-customdlr2stage-ext

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): 0.0000 [0.0000, 0.0000], p=1.000000
- McNemar (correctness): b=0 (A right, B wrong), c=0 (A wrong, B right), n=0, exact p=1.000000

## sensitiveai-vi-stage2

- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): 0.0085 [-0.0030, 0.0196], p=0.147000
- McNemar (correctness): b=94 (A right, B wrong), c=124 (A wrong, B right), n=218, exact p=0.049266
