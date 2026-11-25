# Tech Stack - SensitiveAI

> Tài liệu tóm tắt công nghệ của dự án **SensitiveAI** (phát hiện hate speech / offensive content tiếng Việt và tiếng Anh), dùng để đọc và làm slide thuyết trình.

## 1. Tổng quan dự án

- **Loại dự án:** nghiên cứu ML/NLP + backend moderation service.
- **Bài toán:** phân loại văn bản `HATE / OFFENSIVE / CLEAN`, xử lý dữ liệu mất cân bằng.
- **Phạm vi:** ablation 4 hệ thống trên cùng backbone `mDeBERTa-v3-base`; rule engine; language router; hybrid fusion; FastAPI moderation API.
- **Frontend:** không có trong repo; tương tác qua REST API.
- **Mã nguồn chính:** Python, 49 file / ~8.208 dòng trong `src/`, `scripts/`, `main.py`.

## 2. Ngôn ngữ, runtime và hạ tầng

| Công nghệ | Phiên bản / ghi chú |
|---|---|
| Python | 3.11.x (đã kiểm thử 3.11.9) |
| Hệ điều hành phát triển | Windows 11 |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM |
| CUDA runtime | 12.6 (`torch 2.13.0+cu126`) |
| NVIDIA driver | 617.14 |
| Container | Docker, base image `python:3.11-slim-bookworm` |
| Cấu hình Docker | FastAPI service port mặc định `8001`, 1 worker |

## 3. Core Machine Learning / NLP

| Thư viện | Phiên bản | Vai trò |
|---|---|---|
| PyTorch | 2.13.0+cu126 | Tensor, training loop, CUDA, mixed precision |
| Hugging Face Transformers | 5.14.1 | Backbone encoder, tokenizer, model loading |
| tokenizers | 0.22.2 | Tokenization nhanh |
| sentencepiece | >=0.1.99 | Tokenizer cho một số backbone |
| safetensors | >=0.4 | Lưu/truy cập trọng số model an toàn |
| accelerate | >=0.30 | Hỗ trợ device, mixed precision |
| datasets | 5.0.0 / >=2.18 | Tải và xử lý dataset ViHSD |
| huggingface_hub | >=0.23 | Tải model/dataset từ Hugging Face |
| scikit-learn | 1.8.0 / >=1.3 | Metrics, logistic regression meta-classifier, thống kê |
| scipy | transitive | Kiểm định thống kê (`binomtest`, `chi2`) |

Ghi chú: dự án **không dùng Hugging Face Trainer**, mà dùng training loop tự viết trong `src/training/trainer.py` để hỗ trợ multi-stage, curriculum learning, discriminative learning rates và custom callback.

## 4. Mô hình và backbone

| Mô hình / backbone | Nguồn | Vai trò |
|---|---|---|
| mDeBERTa-v3-base | `microsoft/mdeberta-v3-base` | Backbone chính cho 4 hệ thống (~278M params, hidden size 768) |
| PhoBERT | `vinai/phobert-base-v2` | Baseline đơn ngữ tiếng Việt |
| ViSoBERT | `uitnlp/visobert` | Baseline đơn ngữ tiếng Việt |
| fastText | train local | Embedding lexicon experiments |
| DeBERTaV2Tokenizer / AutoTokenizer | Hugging Face | Tokenization, max length 256 |

Kiến trúc head:
- Standard Head: `AutoModelForSequenceClassification` style (`[CLS]` pooling + linear).
- Custom Head: multi-view pooling (`cls`, `mean`, `max`) + LayerNorm + Fusion MLP + classifier.

## 5. Dữ liệu và nguồn dữ liệu

| Nguồn | Mô tả |
|---|---|
| ViHSD | `uitnlp/vihsd` trên Hugging Face, 33.398 mẫu sau tiền xử lý, 3 nhãn `HATE / OFFENSIVE / CLEAN` |
| Behitek Vietnam Sensitive Words | Lexicon tiếng Việt từ GitHub |
| Blue Eyes Vietnamese Offensive Words | Lexicon offensive + biến thể không dấu / teencode |
| ViCM, Bontori | Dùng cho corpus lexicon / fastText experiments |
| English datasets | Dữ liệu tiếng Anh được chuẩn bị riêng qua scripts 52-54 |

Đặc điểm xử lý dữ liệu:
- Canonical label mapping: `0=HATE, 1=OFFENSIVE, 2=CLEAN`.
- Split cố định `train/validation/test` với SHA-256 fingerprint.
- Tiền xử lý 12 bước, giữ tối đa thông tin, không mặc định xóa URL/mention.
- Lexicon 668+ entry sau prune/verify.

## 6. Backend, API và moderation service

| Công nghệ | Phiên bản | Vai trò |
|---|---|---|
| FastAPI | >=0.110 / Docker pin 0.110.0 | REST API moderation service |
| Uvicorn | >=0.29 / Docker pin 0.29.0 | ASGI server |
| Pydantic | >=2.7 / Docker pin 2.7.4 | Request/response validation |

Endpoints chính:
- `GET /health`
- `POST /api/v1/moderate`
- `POST /api/v1/moderate/batch`

Response includes: `decision`, `score`, `categories`, `matches`, `risk`, `recommendation`, `language`, `message`.

## 7. Rule engine, language routing và fusion

| Thành phần | Công nghệ / thư viện | Ghi chú |
|---|---|---|
| Rule engine | Custom Python + YAML config | Lexical classifier, matcher, normalizer, obfuscation, scoring |
| Language detection | `langid` optional hoặc heuristic tự viết | Fallback dependency-free nếu không cài `langid` |
| Vietnamese tokenization | `underthesea` (script-level) | Dùng trong lexicon experiments, không nằm trong requirements chính |
| Language router | Custom Python + YAML | Chọn rule engine theo `vi/en/mix/multilingual` |
| Hybrid fusion | Custom Python | Weighted fusion, per-channel alpha, threshold tuning |
| Meta-classifier | scikit-learn `LogisticRegression` | Fusion over rule/AI scores |
| Thống kê | scipy.stats | Binomial/McNemar-style analysis |

## 8. Phân tích dữ liệu và trực quan hóa

| Thư viện | Phiên bản | Vai trò |
|---|---|---|
| pandas | 3.0.3 / >=2.0 | DataFrame, CSV, report tables |
| numpy | 2.4.6 / >=1.24 | Tính toán số |
| matplotlib | 3.11.1 / >=3.7 | Biểu đồ training/evaluation |
| seaborn | 0.13.2 / >=0.12 | Style và biểu đồ thống kê |
| tabulate | >=0.9 | Markdown tables trong report |

## 9. Cấu hình, lưu trữ và định dạng file

| Loại | Công nghệ / định dạng |
|---|---|
| Config | YAML (`PyYAML >=6.0`) |
| Dữ liệu | CSV |
| Metadata / metrics | JSON |
| Model weights | Safetensors, PyTorch `.pt` checkpoints |
| Hình ảnh | PNG, PDF |
| Paper/report | LaTeX (LNCS style), XeLaTeX |

Cấu trúc output chính:
- `reports/<model>/run_00N/`
- `experiments/<model>/run_00N/`
- `evaluation/comparison/`
- `FINALIZATION/source/iccies2027/`

## 10. Công cụ phát triển và kiểm thử

| Công cụ | Vai trò |
|---|---|
| Git | Quản lý mã nguồn |
| Docker | Đóng gói moderation service |
| pytest | Kiểm thử trong `FINALIZATION/tests/` |
| VS Code | IDE, LaTeX Workshop |
| XeLaTeX | Build paper ICCIES2027 |
| Python scripts 00-68 | Pipeline dữ liệu, train, eval, fusion, phân tích |

## 11. Phụ thuộc tùy chọn / script-level

| Thư viện | Trạng thái | Ghi chú |
|---|---|---|
| `langid >=1.1.6` | Optional trong requirements | Language detection; fallback heuristic nếu không cài |
| `bitsandbytes >=0.43` | Optional trong requirements | 8-bit optimizer cho GPU 8 GB |
| `wandb >=0.16` | Optional trong requirements | Logging experiment |
| `fasttext` | Script-level, không trong requirements | Train embedding lexicon |
| `underthesea` | Script-level, không trong requirements | Vietnamese word tokenization |
| `scipy` | Transitive dependency | Thống kê |

## 12. Gợi ý cho slide thuyết trình

- **Stack cốt lõi:** Python 3.11 + PyTorch + Hugging Face Transformers.
- **Backbone chính:** `microsoft/mdeberta-v3-base` cho cả 4 hệ thống ablation.
- **API serving:** FastAPI + Uvicorn + Pydantic, Dockerized, port 8001.
- **Dữ liệu:** ViHSD 33.398 mẫu, split cố định có SHA-256 fingerprint.
- **Hybrid:** Rule engine + language router + AI model + fusion.
- **Môi trường benchmark:** Windows 11, RTX 4060 8 GB, CUDA 12.6.
- **Điểm khác biệt:** custom training loop thay vì Hugging Face Trainer, hỗ trợ curriculum/multi-stage/DLR.
- **Không có frontend trong repo**, chỉ backend REST API và pipeline nghiên cứu.
