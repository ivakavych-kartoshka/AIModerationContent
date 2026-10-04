SensitiveAI — Research, Training & Evaluation Plan

1. Mục tiêu
Xây dựng hệ thống AI phát hiện nội dung có khả năng vi phạm tiêu chuẩn cộng đồng bằng tiếng Việt, sử dụng:
Base model:
microsoft/mdeberta-v3-base
Dataset:
ViHSD — Vietnamese Hate Speech Detection
https://github.com/sonlam1102/vihsd
Bài toán gồm 3 nhãn:
HATE
OFFENSIVE
CLEAN
Mục tiêu là train và so sánh 4 model:
1. sensitiveai-vi
2. sensitiveai-vi-custom
3. sensitiveai-vi-customdlr2stage
4. sensitiveai-vi-custom-curriculum
Tất cả 4 model phải sử dụng cùng dataset split, cùng label mapping và cùng test set để đảm bảo kết quả so sánh công bằng.

2. Dataset
2.1. Vietnamese Dataset
Sử dụng ViHSD với 3 nhãn:
HATE
OFFENSIVE
CLEAN
Mapping cố định:
HATE       = 0
OFFENSIVE  = 1
CLEAN      = 2
Không được thay đổi mapping giữa các model.

3. Preprocessing
Áp dụng preprocessing thống nhất cho toàn bộ dataset:
•	Loại bỏ dòng không có text.
•	Xử lý null/NaN.
•	lower().
•	Chuẩn hóa khoảng trắng.
•	Chuẩn hóa newline/tab nếu cần.
•	Giữ nguyên tiếng Việt có dấu.
•	Không thực hiện preprocessing làm mất thông tin quan trọng.
•	Tokenize bằng tokenizer của microsoft/mdeberta-v3-base.
Quy tắc quan trọng
Không được xóa test samples.
Test set phải được giữ nguyên số lượng sample sau khi split.
Preprocessing có thể biến đổi nội dung text, nhưng không được loại bỏ test sample chỉ vì preprocessing.

4. Dataset Split
Tạo split cố định:
data/
├── raw/
├── processed/
└── splits/
    ├── train.csv
    ├── validation.csv
    └── test.csv
Tất cả 4 model sử dụng chính xác cùng:
train.csv
validation.csv
test.csv
Sử dụng stratified split theo:
HATE
OFFENSIVE
CLEAN
Sử dụng random seed cố định.
Không được split lại dataset cho từng model.

5. Data Leakage Prevention
Quy trình:
TRAIN
  ↓
Training

VALIDATION
  ↓
Model selection
Hyperparameter selection
Threshold selection
Early stopping

TEST
  ↓
Final evaluation
Không được:
•	train trên test;
•	chọn best model bằng test;
•	chọn threshold bằng test;
•	tune hyperparameter bằng test;
•	dùng test để quyết định training configuration.

6. Model 1 — sensitiveai-vi
Mục đích
Baseline model.
Architecture:
mDeBERTa-v3-base
       ↓
Standard Classification Head
       ↓
HATE / OFFENSIVE / CLEAN
Train trực tiếp trên ViHSD.
Sử dụng 3-class classification.
Nếu mỗi sample chỉ có một label:
CrossEntropyLoss
Model này được dùng làm baseline cho các model còn lại.

7. Model 2 — sensitiveai-vi-custom
Mục đích
Đánh giá ảnh hưởng của Custom Classification Head.
Architecture:
sensitiveai-vi       
↓
Custom Classification Head
       ↓
HATE / OFFENSIVE / CLEAN
Custom Head phải được định nghĩa rõ trong source code.
Không thay đổi dataset hoặc dataset split.
Nếu sử dụng checkpoint sensitive-v1 cũ, phải ghi rõ checkpoint/version/path trong configuration.
Nếu không có checkpoint đó, sử dụng:
sensitiveai-vi làm encoder initialization.

8. Model 3 — sensitiveai-vi-customdlr2stage
Model này sử dụng:
•	Custom Head
•	Discriminative Learning Rates
•	Two-stage Training
•	Callback
•	Class Weight
•	Label-specific Class Weight
•	Focal Loss
•	Label-specific Threshold

8.1. Custom Head
Sử dụng cùng Custom Head với:
sensitiveai-vi-custom
để đảm bảo so sánh hợp lý.

8.2. Discriminative Learning Rates
Không sử dụng cùng learning rate cho toàn bộ model.
Ví dụ:
Lower Transformer layers
        ↓
Low learning rate

Upper Transformer layers
        ↓
Medium learning rate

Classification Head
        ↓
Higher learning rate
Configuration phải lưu riêng:
encoder_lr
layer_lr
head_lr

8.3. Two-stage Training
Stage 1
Fine-tune model với configuration ban đầu.
Mục tiêu:
•	ổn định pretrained representation;
•	học task-specific representation;
•	tránh cập nhật encoder quá mạnh.
Stage 2
Tiếp tục từ checkpoint của Stage 1.
Có thể sử dụng:
•	learning rate thấp hơn;
•	DLR;
•	Focal Loss;
•	class-specific weighting.
Việc chuyển Stage 1 → Stage 2 phải được quản lý bằng Callback/Training Controller.
Mỗi stage phải được ghi log riêng.

9. Class Weight
Tính class weight từ training set.
Ví dụ:
HATE       → weight_hate
OFFENSIVE  → weight_offensive
CLEAN      → weight_clean
Không được tính weight từ validation hoặc test.

10. Label-specific Class Weight
Cho phép mỗi class có weight riêng:
HATE       → w_hate
OFFENSIVE  → w_offensive
CLEAN      → w_clean
Weight phải được lưu vào experiment configuration.
Nếu weight được điều chỉnh dựa trên validation performance thì phải ghi lại quá trình điều chỉnh.

11. Focal Loss
Sử dụng Focal Loss để tập trung vào các sample khó:
Easy samples
    ↓
lower contribution

Hard / misclassified samples
    ↓
higher contribution
Lưu các hyperparameter:
gamma
alpha

12. Label-specific Threshold
Nếu sử dụng threshold riêng cho từng label, threshold phải được tìm trên validation set.
Ví dụ:
HATE       → threshold_hate
OFFENSIVE  → threshold_offensive
CLEAN      → threshold_clean
Sau khi chọn:
Validation
    ↓
Threshold optimization
    ↓
Freeze threshold
    ↓
Test evaluation
Lưu:
thresholds.json
Không tối ưu threshold trên test set.

13. Model 4 — sensitiveai-vi-custom-curriculum
Sử dụng Custom Head và curriculum training gồm 3 stage.

Stage 1 — OFFENSIVE-focused
Mục tiêu:
Tập trung học đặc trưng của OFFENSIVE.
Concept:
OFFENSIVE → positive
HATE      → negative
CLEAN     → negative
Vẫn sử dụng toàn bộ dataset.
Không được chỉ train bằng OFFENSIVE samples.
Lưu checkpoint Stage 1.

Stage 2 — HATE-focused
Tiếp tục từ Stage 1.
Concept:
HATE      → positive
OFFENSIVE → negative
CLEAN     → negative
Vẫn sử dụng toàn bộ dataset.
Lưu checkpoint Stage 2.

Stage 3 — Final 3-class training
Tiếp tục từ Stage 2.
Train trên:
HATE
OFFENSIVE
CLEAN
Mục tiêu cuối cùng:
P(HATE)
P(OFFENSIVE)
P(CLEAN)
Model cuối cùng:
sensitiveai-vi-custom-curriculum

14. Training Configuration
Mỗi model có configuration riêng:
configs/
├── sensitiveai-vi.yaml
├── sensitiveai-vi-custom.yaml
├── sensitiveai-vi-customdlr2stage.yaml
└── sensitiveai-vi-custom-curriculum.yaml
Configuration phải lưu:
model_name
base_model
dataset
seed
max_length
batch_size
gradient_accumulation
epochs
learning_rate
weight_decay
warmup_ratio
optimizer
loss
early_stopping
Đối với DLR:
encoder_lr
layer_lr
head_lr
stage_1_config
stage_2_config
Đối với Curriculum:
stage_1_config
stage_2_config
stage_3_config

15. Research Folder Structure
SensitiveAI/
│
├── data/
│   ├── raw/
│   │   └── vihsd/
│   │
│   ├── processed/
│   │   └── vihsd/
│   │
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── losses/
│   ├── callbacks/
│   ├── training/
│   └── evaluation/
│
├── configs/
│   ├── sensitiveai-vi.yaml
│   ├── sensitiveai-vi-custom.yaml
│   ├── sensitiveai-vi-customdlr2stage.yaml
│   └── sensitiveai-vi-custom-curriculum.yaml
│
├── experiments/
│   ├── sensitiveai-vi/
│   ├── sensitiveai-vi-custom/
│   ├── sensitiveai-vi-customdlr2stage/
│   └── sensitiveai-vi-custom-curriculum/
│
├── checkpoints/
│
├── reports/
│   ├── sensitiveai-vi/
│   ├── sensitiveai-vi-custom/
│   ├── sensitiveai-vi-customdlr2stage/
│   └── sensitiveai-vi-custom-curriculum/
│
├── evaluation/
│   ├── per_model/
│   ├── comparison/
│   ├── plots/
│   └── logs/
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   ├── tables/
│   └── supplementary/
│
├── scripts/
│
└── README.md

16. Report cho từng model
Mỗi model có một thư mục riêng.
Ví dụ:
reports/
└── sensitiveai-vi/
    └── run_001/
        ├── config.yaml
        ├── model_summary.txt
        ├── training_log.csv
        ├── best_model/
        ├── confusion_matrix.png
        ├── classification_report.txt
        ├── pr_curve.png
        ├── roc_curve.png
        ├── val_accuracy_curve.png
        └── val_loss_curve.png
Tương tự cho 4 model.

17. Training Log
Mỗi run phải lưu:
training_log.csv
Tối thiểu:
epoch
accuracy
learning_rate
loss
val_accuracy
val_loss
Nên bổ sung:
val_macro_f1
val_weighted_f1
Đối với DLR:
encoder_lr
head_lr
Đối với Curriculum:
stage
epoch
loss
accuracy
learning_rate
val_loss
val_accuracy
val_macro_f1

18. Validation Accuracy Curve
Đây là phần giống hình đầu tiên bạn gửi.
Sau mỗi model training, tạo:
val_accuracy_curve.png
Biểu đồ:
X-axis → Epoch
Y-axis → Validation Accuracy
Ví dụ:
Validation Accuracy
        │
  95% ──┐             ╭────
        │          ╭───╯
  90% ──┤      ╭───╯
        │   ╭──╯
  85% ──┤───╯
        │
        └────────────────────
          Epoch
Đường biểu diễn phải lấy trực tiếp từ:
training_log.csv

19. Validation Loss Curve
Tạo:
val_loss_curve.png
Giống hình thứ hai bạn gửi.
Validation Loss
        │
  1.2 ──╮
        │ ╲
  0.8 ──┤  ╲
        │    ╲
  0.4 ──┤      ╲────
        │
        └────────────────────
          Epoch
X-axis:
Epoch
Y-axis:
Validation Loss
Mục đích để theo dõi:
•	convergence;
•	overfitting;
•	training stability;
•	epoch đạt performance tốt nhất.

20. Confusion Matrix
Mỗi model phải tạo một confusion matrix 3×3 riêng.
Ví dụ:
                 Predicted
             HATE OFFENSIVE CLEAN
Actual HATE      XX     XX      XX
       OFFENSIVE XX     XX      XX
       CLEAN     XX     XX      XX
Tên class bắt buộc phải hiển thị:
HATE
OFFENSIVE
CLEAN
Không chỉ sử dụng:
0 / 1 / 2
Nên lưu cả:
confusion_matrix_raw.png
confusion_matrix_normalized.png
nếu có thể.

21. PR Curve — từng model
Mỗi model phải có Precision-Recall Curve riêng.
Ví dụ:
reports/
├── sensitiveai-vi/
│   └── run_001/
│       └── pr_curve.png
│
├── sensitiveai-vi-custom/
│   └── run_001/
│       └── pr_curve.png
│
...
Vì đây là bài toán 3-class, PR curve phải được tính theo One-vs-Rest:
HATE vs NOT-HATE

OFFENSIVE vs NOT-OFFENSIVE

CLEAN vs NOT-CLEAN
Biểu đồ của từng model nên hiển thị:
HATE
OFFENSIVE
CLEAN
và có thể thêm:
Macro Average
Micro Average
Nếu có tính Average Precision, ghi:
AP_HATE
AP_OFFENSIVE
AP_CLEAN
Macro AP
Micro AP

22. ROC Curve — từng model
Mỗi model phải có:
roc_curve.png
Tương tự PR:
HATE vs Rest

OFFENSIVE vs Rest

CLEAN vs Rest
Có thể hiển thị:
HATE
OFFENSIVE
CLEAN
Macro Average
Micro Average
Lưu:
ROC-AUC per class
Macro ROC-AUC
Micro ROC-AUC

23. Tổng hợp PR Curve cho 4 model
Sau khi train xong cả 4 model, chạy evaluation pipeline tổng.
Tạo:
evaluation/comparison/
└── pr_curve_all_models.png
Biểu đồ so sánh:
sensitiveai-vi
sensitiveai-vi-custom
sensitiveai-vi-customdlr2stage
sensitiveai-vi-custom-curriculum
Để biểu đồ không quá rối, nên có hai dạng:
PR comparison — Macro
Một đường cho mỗi model:
Model 1
Model 2
Model 3
Model 4
PR comparison — Per Class
Tạo riêng:
pr_hate_all_models.png
pr_offensive_all_models.png
pr_clean_all_models.png
Như vậy có thể trực tiếp so sánh 4 model trên từng class.

24. Tổng hợp ROC Curve cho 4 model
Tương tự PR.
Tạo:
evaluation/comparison/
├── roc_curve_all_models.png
├── roc_hate_all_models.png
├── roc_offensive_all_models.png
└── roc_clean_all_models.png
Mỗi biểu đồ cho phép so sánh:
sensitiveai-vi
sensitiveai-vi-custom
sensitiveai-vi-customdlr2stage
sensitiveai-vi-custom-curriculum

25. Tổng hợp Validation Accuracy của 4 model
Ngoài PR/ROC, cần có biểu đồ giống hình đầu tiên bạn gửi.
Tạo:
evaluation/comparison/
└── validation_accuracy_all_models.png
Biểu đồ:
X-axis → Epoch
Y-axis → Validation Accuracy
Có 4 đường:
sensitiveai-vi
sensitiveai-vi-custom
sensitiveai-vi-customdlr2stage
sensitiveai-vi-custom-curriculum
Mỗi đường phải lấy dữ liệu từ training_log.csv.

26. Tổng hợp Validation Loss của 4 model
Tạo:
evaluation/comparison/
└── validation_loss_all_models.png
Biểu đồ:
X-axis → Epoch
Y-axis → Validation Loss
Có 4 đường tương ứng 4 model.
Mục đích:
•	so sánh tốc độ convergence;
•	phát hiện overfitting;
•	so sánh training stability;
•	xem model nào đạt validation loss thấp hơn theo epoch.

27. Model Summary
Mỗi model phải ghi:
model_summary.txt
Bao gồm:
Model Name
Base Model

Total Parameters
Trainable Parameters
Non-trainable Parameters
Estimated Parameter Memory

Training Samples
Validation Samples
Test Samples

Best Epoch
Training Time

Average Inference Latency
FPS

GPU
CUDA Version
PyTorch Version
Transformers Version
Latency và FPS phải được benchmark trên cùng hardware, cùng batch size và cùng inference configuration.

28. Classification Report
Mỗi model:
Best_model_classification_report.txt
Bao gồm:
                precision
                recall
                f1-score
                support

HATE
OFFENSIVE
CLEAN

accuracy
macro avg
weighted avg

29. Evaluation Result
Sau khi chạy xong 4 model:
evaluation/
├── per_model/
│   ├── sensitiveai-vi/
│   ├── sensitiveai-vi-custom/
│   ├── sensitiveai-vi-customdlr2stage/
│   └── sensitiveai-vi-custom-curriculum/
│
├── comparison/
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   ├── pr_curve_all_models.png
│   ├── roc_curve_all_models.png
│   ├── pr_hate_all_models.png
│   ├── pr_offensive_all_models.png
│   ├── pr_clean_all_models.png
│   ├── roc_hate_all_models.png
│   ├── roc_offensive_all_models.png
│   ├── roc_clean_all_models.png
│   ├── validation_accuracy_all_models.png
│   └── validation_loss_all_models.png
│
├── plots/
└── logs/

30. Model Comparison Table
Tạo:
evaluation/comparison/model_comparison.csv
Các cột:
Model
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1

HATE Precision
HATE Recall
HATE F1

OFFENSIVE Precision
OFFENSIVE Recall
OFFENSIVE F1

CLEAN Precision
CLEAN Recall
CLEAN F1

HATE ROC-AUC
OFFENSIVE ROC-AUC
CLEAN ROC-AUC
Macro ROC-AUC

HATE AP
OFFENSIVE AP
CLEAN AP
Macro AP

Parameters
Latency
FPS
Đây sẽ là bảng dữ liệu chính để viết phần Results của paper.

31. Evaluation Log
Tạo:
evaluation/logs/evaluation_log.csv
Ghi:
timestamp
model
checkpoint
dataset
split
accuracy
macro_f1
weighted_f1
roc_auc_macro
average_precision_macro
latency
fps

32. Paper-ready Output
Các kết quả evaluation phải có thể sử dụng trực tiếp cho paper.
paper/
├── main.tex
├── references.bib
│
├── figures/
│   ├── architecture.png
│   ├── validation_accuracy_all_models.png
│   ├── validation_loss_all_models.png
│   ├── pr_curve_all_models.png
│   ├── roc_curve_all_models.png
│   ├── pr_hate_all_models.png
│   ├── pr_offensive_all_models.png
│   └── pr_clean_all_models.png
│
├── tables/
│   ├── model_comparison.tex
│   ├── classification_results.tex
│   └── ablation_results.tex
│
└── supplementary/
Không copy số liệu thủ công vào LaTeX.
Evaluation script nên tự sinh .csv và .tex.

33. Tổng pipeline
Toàn bộ workflow:
                    ViHSD
                      │
                      ▼
                Preprocessing
                      │
                      ▼
              Fixed Data Split
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
        TRAIN     VALIDATION     TEST
          │           │           │
          └──────┬────┘           │
                 │                │
       ┌─────────┴─────────┐      │
       │                   │      │
       ▼                   ▼      │
 sensitiveai-vi      custom models
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
          Custom       Custom DLR       Curriculum
           Head        2-stage           3-stage
             │             │              │
             └─────────────┴──────────────┘
                           │
                           ▼
                    Best Checkpoint
                           │
                           ▼
                    Evaluation
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    Confusion Matrix    PR / ROC       Accuracy/Loss
       3 × 3            Curves            Curves
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 4-Model Comparison
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          PR/ROC       Accuracy/Loss   Metrics Table
         Comparison      Comparison
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Paper-ready Results
34. Bộ output cuối cùng
Sau khi hoàn thành, mỗi model sẽ có:
✓ Best model
✓ Model summary
✓ Training log
✓ 3×3 Confusion Matrix
✓ Normalized Confusion Matrix
✓ Classification Report
✓ PR Curve
✓ ROC Curve
✓ Validation Accuracy Curve
✓ Validation Loss Curve
Và sau khi hoàn thành cả 4 model:
✓ 4-model PR comparison
✓ 4-model ROC comparison
✓ HATE PR comparison
✓ OFFENSIVE PR comparison
✓ CLEAN PR comparison
✓ HATE ROC comparison
✓ OFFENSIVE ROC comparison
✓ CLEAN ROC comparison
✓ 4-model Validation Accuracy comparison
✓ 4-model Validation Loss comparison
✓ Model comparison CSV
✓ Model comparison table
✓ Paper-ready figures
✓ Paper-ready LaTeX tables

