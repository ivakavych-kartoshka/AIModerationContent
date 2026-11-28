# End-to-end language-aware pipeline - mixed sample

- input sample: 2000 texts (EN 1000 / VI 1000)
- fusion alpha: 0.7
- decision: argmax (no extra thresholds)
- routing time: 12021.5 ms total (6.01 ms/text)
- AI inference time: 8961.6 ms total (4.48 ms/text, 2 batches)

## Aggregate (fused scores -> final decision)

- accuracy : 0.8715
- macro_f1: 0.7854
- weighted_f1: 0.8724

## Per-language breakdown

- vi: n=1000 acc=0.8690 macro_f1=0.6702
- en: n=1000 acc=0.8740 macro_f1=0.7490