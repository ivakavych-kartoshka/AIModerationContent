# End-to-end language-aware pipeline - mixed sample

- input sample: 1000 texts (EN 500 / VI 500)
- fusion alpha: 0.7
- decision: argmax (no extra thresholds)
- routing time: 6849.9 ms total (6.85 ms/text)
- AI inference time: 4588.6 ms total (4.59 ms/text, 2 batches)

## Aggregate (fused scores -> final decision)

- accuracy : 0.8700
- macro_f1: 0.7829
- weighted_f1: 0.8701

## Per-language breakdown

- vi: n=500 acc=0.8600 macro_f1=0.6500
- en: n=500 acc=0.8800 macro_f1=0.7430