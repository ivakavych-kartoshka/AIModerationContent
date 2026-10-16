# Cross-split duplication and de-leaked robustness

- test comments: 6,680
- test comments with an exact duplicate in train or validation: 895 (13.4%)
- de-leaked test subset: 5,785
- distinct strings in >1 split: 1,188 (label-conflicting: 118)
- pairwise distinct-string overlaps: {'train_validation': 326, 'train_test': 794, 'validation_test': 94}

| System | Full macro-F1 | De-leaked macro-F1 | Leaked-only macro-F1 | Full acc. | De-leaked acc. |
|---|---|---|---|---|---|
| M1 | 0.6545 | 0.6533 | 0.6592 | 0.8734 | 0.8681 |
| M2 | 0.6575 | 0.6576 | 0.6489 | 0.8762 | 0.8719 |
| M3 | 0.6638 | 0.6654 | 0.6425 | 0.8754 | 0.8721 |
| M4 | 0.6590 | 0.6601 | 0.6433 | 0.8731 | 0.8698 |
| M5 | 0.6536 | 0.6537 | 0.6447 | 0.8783 | 0.8748 |
| M6 | 0.6604 | 0.6606 | 0.6512 | 0.8633 | 0.8588 |
