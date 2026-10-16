# Model comparison (ViHSD test set)

Latency = milliseconds per sample; FPS = samples per second. Both measured on the same GPU with the same tokenizer, max_length and batch size.

| Model                 |   Accuracy |   Macro F1 |   Weighted F1 |   HATE F1 |   OFFENSIVE F1 |   CLEAN F1 |   Macro ROC-AUC |   Macro AP |   Parameters |   Latency |    FPS |
|:----------------------|-----------:|-----------:|--------------:|----------:|---------------:|-----------:|----------------:|-----------:|-------------:|----------:|-------:|
| sensitiveai-vi        |     0.8734 |     0.6545 |        0.8705 |    0.5864 |         0.4367 |     0.9405 |          0.8873 |     0.6599 |  278,811,651 |   28.1451 | 35.53  |
| sensitiveai-vi-stage2 |     0.869  |     0.6599 |        0.8699 |    0.6091 |         0.4336 |     0.9372 |          0.891  |     0.6484 |  278,811,651 |   11.583  | 86.333 |

## Full table

| Model                 |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   HATE Precision |   HATE Recall |   HATE F1 |   OFFENSIVE Precision |   OFFENSIVE Recall |   OFFENSIVE F1 |   CLEAN Precision |   CLEAN Recall |   CLEAN F1 |   HATE ROC-AUC |   OFFENSIVE ROC-AUC |   CLEAN ROC-AUC |   Macro ROC-AUC |   HATE AP |   OFFENSIVE AP |   CLEAN AP |   Macro AP |   Parameters |   Latency |    FPS | Decision Rule   |   Best Epoch |   Training Time (s) |   Micro ROC-AUC |   Test Samples |
|:----------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-----------------:|--------------:|----------:|----------------------:|-------------------:|---------------:|------------------:|---------------:|-----------:|---------------:|--------------------:|----------------:|----------------:|----------:|---------------:|-----------:|-----------:|-------------:|----------:|-------:|:----------------|-------------:|--------------------:|----------------:|---------------:|
| sensitiveai-vi        |   0.873353 |          0.670103 |       0.64223  |   0.654525 |      0.87052  |         0.644599 |      0.537791 |  0.586371 |              0.434298 |           0.439189 |       0.43673  |          0.931412 |       0.949712 |   0.940473 |       0.894612 |            0.868586 |        0.898717 |        0.887305 |  0.598154 |       0.411908 |   0.969779 |   0.659947 |    278811651 |   28.1451 | 35.53  | argmax          |            9 |           10203.8   |        0.963481 |           6680 |
| sensitiveai-vi-stage2 |   0.869012 |          0.657419 |       0.662774 |   0.65993  |      0.869897 |         0.613569 |      0.604651 |  0.609078 |              0.419831 |           0.448198 |       0.433551 |          0.938857 |       0.935472 |   0.937161 |       0.900461 |            0.865624 |        0.906853 |        0.890979 |  0.581045 |       0.390298 |   0.973715 |   0.648353 |    278811651 |   11.583  | 86.333 | thresholds      |            5 |             701.791 |        0.957868 |           6680 |
