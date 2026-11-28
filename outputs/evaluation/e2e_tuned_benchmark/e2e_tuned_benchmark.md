# End-to-end tuned benchmark

- **ai_argmax_e2e**: acc 0.8714, macro-F1 0.7792, ECE 0.1006
- **ai_thresholds_e2e**: acc 0.8719, macro-F1 0.7803, ECE 0.1011
- **hybrid_e2e**: acc 0.8728, macro-F1 0.7787, ECE 0.0779
- **ai_argmax_oracle**: acc 0.8812, macro-F1 0.7898, ECE 0.0919
- **ai_thresholds_oracle**: acc 0.8817, macro-F1 0.7910, ECE 0.0925
- **hybrid_oracle**: acc 0.8826, macro-F1 0.7896, ECE 0.0698

## Per language
- ai_argmax_e2e / vi: acc 0.8713, macro-F1 0.6602, H 0.6177, O 0.4252, C 0.9376
- ai_argmax_e2e / en: acc 0.8717, macro-F1 0.7273, H 0.4407, O 0.9192, C 0.8218
- ai_thresholds_e2e / vi: acc 0.8720, macro-F1 0.6612, H 0.6173, O 0.4282, C 0.9379
- ai_thresholds_e2e / en: acc 0.8717, macro-F1 0.7324, H 0.4560, O 0.9196, C 0.8215
- hybrid_e2e / vi: acc 0.8732, macro-F1 0.6603, H 0.6098, O 0.4322, C 0.9388
- hybrid_e2e / en: acc 0.8717, macro-F1 0.7313, H 0.4525, O 0.9196, C 0.8218
- ai_argmax_oracle / vi: acc 0.8711, macro-F1 0.6608, H 0.6197, O 0.4251, C 0.9375
- ai_argmax_oracle / en: acc 0.9084, macro-F1 0.7676, H 0.4505, O 0.9439, C 0.9083
- ai_thresholds_oracle / vi: acc 0.8719, macro-F1 0.6617, H 0.6194, O 0.4281, C 0.9378
- ai_thresholds_oracle / en: acc 0.9084, macro-F1 0.7726, H 0.4656, O 0.9443, C 0.9081
- hybrid_oracle / vi: acc 0.8731, macro-F1 0.6609, H 0.6119, O 0.4320, C 0.9387
- hybrid_oracle / en: acc 0.9084, macro-F1 0.7726, H 0.4656, O 0.9443, C 0.9081

## Significance
- hybrid_vs_ai_argmax_e2e: delta -0.0004 [-0.0045,+0.0036], p=0.8412; McNemar 44/31, p=0.1654
- hybrid_vs_ai_argmax_oracle: delta -0.0002 [-0.0042,+0.0037], p=0.9134; McNemar 43/30, p=0.1597
