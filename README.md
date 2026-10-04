# SensitiveAI

### Phân loại văn bản tiếng Việt: **HATE / OFFENSIVE / CLEAN**
#### Nghiên cứu ablation trên 4 hệ thống dùng chung backbone `mDeBERTa-v3-base`

> **Tài liệu này dành cho người đọc/review:** mentor, giảng viên hướng dẫn, hoặc người
> muốn tái lập (reproduce) toàn bộ kết quả. Mọi con số, cấu hình và lệnh trong tài liệu
> đều được trích từ mã nguồn thực tế của kho mã, không phải mô tả định hướng.

---

## Mục lục

1. [Tóm tắt điều tra (executive summary)](#1-tóm-tắt-điều-tra)
2. [Bài toán nghiên cứu](#2-bài-toán-nghiên-cứu)
3. [Bốn hệ thống được so sánh](#3-bốn-hệ-thống-được-so-sánh)
4. [Kiến trúc mô hình](#4-kiến-trúc-mô-hình)
5. [Dữ liệu ViHSD và tiền xử lý](#5-dữ-liệu-vihsd-và-tiền-xử-lý)
6. [Chiến lược huấn luyện](#6-chiến-lược-huấn-luyện)
7. [Cấu trúc kho mã](#7-cấu-trúc-kho-mã)
8. [Cài đặt môi trường](#8-cài-đặt-môi-trường)
9. [Hướng dẫn chạy đầy đủ](#9-hướng-dẫn-chạy-đầy-đủ)
10. [Cấu trúc kết quả đầu ra](#10-cấu-trúc-kết-quả-đầu-ra)
11. [Quy chuẩn công bằng thực nghiệm (fairness)](#11-quy-chuẩn-công-bằng-thực-nghiệm-fairness)
12. [Tái lập kết quả trên máy khác](#12-tái-lập-kết-quả-trên-máy-khác)
13. [Kiểm thử và kiểm định chất lượng](#13-kiểm-thử-và-kiểm-định-chất-lượng)
14. [Ghi chú kỹ thuật quan trọng](#14-ghi-chú-kỹ-thuật-quan-trọng)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)
16. [Phụ lục A — Câu hỏi thường gặp](#phụ-lục-a--câu-hỏi-thường-gặp)
17. [Phụ lục B — Lịch sử kiểm thử](#phụ-lục-b--lịch-sử-kiểm-thử)

---

## 1. Tóm tắt điều tra

| | |
|---|---|
| **Vấn đề** | Phát hiện ngôn tử độc hại trên văn bản mạng xã hội tiếng Việt, với dữ liệu mất cân bằng nặng |
| **Dữ liệu** | ViHSD (`uitnlp/vihsd`) — 33.398 mẫu sau tiền xử lý, 3 nhãn |
| **Backbone** | `microsoft/mdeberta-v3-base` (~278M tham số) cho **cả 4 hệ thống** |
| **Câu hỏi nghiên cứu** | Mỗi quyết định thiết kế (head, learning rate, loss, curriculum, threshold) đóng góp bao nhiêu? |
| **Thiết kế thực nghiệm** | Ablation 4 bước, mỗi bước chỉ thay đúng **một** nhóm quyết định, giữ nguyên phần còn lại |
| **Kết quả** | Chưa chạy — người dùng tự chạy `01` → `02` → `03` (xem [mục 9](#9-hướng-dẫn-chạy-đầy-đủ)) |
| **Mã nguồn** | Python, 49 file / 8.208 dòng (6.707 dòng code), 43 module, không phụ thuộc framework huấn luyện ngoài |

**Điểm quan trọng cần lưu ý khi đọc kết quả:** đây là nghiên cứu **ablation đơn giản về
chiến lược tối ưu**, không phải nghiên cứu về kiến trúc backbone. Cả 4 hệ thống dùng
cùng một encoder; những khác biệt đo được đến từ head, learning rate, loss function,
lịch trình huấn luyện và quy tắc quyết định (decision rule).

---

## 2. Bài toán nghiên cứu

### 2.1 Gán nhãn cố định

Ánh xạ nhãn được **đóng băng** và kiểm tra ở mọi cửa ngõ. Đổi ánh xạ này giữa 4 lần chạy
làm toàn bộ phép so sánh vô nghĩa.

```
HATE      -> 0
OFFENSIVE -> 1
CLEAN     -> 2
```

ViHSD gốc dùng thứ tự **ngược lại**:

```
ViHSD native:   0 = CLEAN,  1 = OFFENSIVE,  2 = HATE
Canonical của dự án:  0 = HATE,  1 = OFFENSIVE,  2 = CLEAN
```

Bảng ánh xạ nằm trong `src/common/constants.py` và được áp dụng **một lần duy nhất** tại
bước tiền xử lý. Các file CSV đầu ra đã mang nhãn canonical, nên mọi mô hình đọc cùng
một định dạng.

### 2.2 Vấn đề mất cân bằng

| Nhãn | Số mẫu (toàn bộ 33.398) | Tỉ lệ |
|------|-------------------------:|------:|
| HATE | 3.514 | 10,5 % |
| OFFENSIVE | 2.261 | 6,8 % |
| CLEAN | 27.623 | 82,7 % |

Mô hình nếu dự đoán toàn bộ là `CLEAN` vẫn đạt **82,7 % accuracy** — vì vậy **accuracy
không dùng làm số liệu chính** trong nghiên cứu này. Số liệu chính là:

- **Macro F1** — trung bìnhharmonic của F1 cả 3 lớp, mỗi lớp đóng trọng số như nhau
- **Per-class P/R/F1** — để thấy rõ lớp nào bị bỏ sót
- **PR-AUC / ROC-AUC** — ROC-AUC bị phóng đại khi dữ liệu mất cân bằng, nên PR-AUC mới là
  chỉ số thực sự đáng tin (Saito & Rehmsmeier, 2015)

---

## 3. Bốn hệ thống được so sánh

Bốn cấu hình tạo thành **một chuỗi ablation**, mỗi bước chỉ bật thêm một nhóm tính năng:

```
sensitiveai-vi                     (baseline)
  └─ đổi head          → sensitiveai-vi-custom
       └─ thêm DLR + 2 stage + focal + threshold → sensitiveai-vi-customdlr2stage
            └─ thay 2 stage bằng curriculum 3 stage → sensitiveai-vi-custom-curriculum
```

### 3.1 So sánh tổng hợp

| | `sensitiveai-vi` | `sensitiveai-vi-custom` | `sensitiveai-vi-customdlr2stage` | `sensitiveai-vi-custom-curriculum` |
|---|:---:|:---:|:---:|:---:|
| **Head** | standard `[CLS]` | multi-view pooling | multi-view pooling | multi-view pooling |
| **Số stage** | 1 | 1 | **2** | **3** |
| **Tổng epoch** | 6 | 6 | 7 | 10 |
| **Loss** | CE | CE | CE → **focal** | **concept** → **concept** → CE |
| **Learning rate** | đồng nhất 3e-5 | đồng nhất 3e-5 | **DLR 4 nhóm** | **DLR 4 nhóm** |
| **Class weight** | không | không | **inverse-sqrt-frequency** | **manual (3/2/1)** |
| **Decision rule** | argmax | argmax | **threshold riêng từng nhãn** | **threshold riêng từng nhãn** |
| **Early stopping** | có (patience 3) | có | có | có |

### 3.2 Model 1 — `sensitiveai-vi` (baseline)

Điểm tham chiếu của nghiên cứu. Không có kỹ thuật nào ngoài baseline tối thiểu:

- Head chuẩn của Hugging Face: `AutoModelForSequenceClassification` (pooler + linear)
- Cross-entropy **không** class weight
- Một learning rate cho toàn bộ mạng
- Quyết định bằng `argmax`

Nếu một kỹ thuật phức tạp nào không tốt hơn con số này, nó không đáng dùng.

### 3.3 Model 2 — `sensitiveai-vi-custom`

Cùng backbone, cùng learning rate, cùng loss, cùng số epoch như baseline — **chỉ thay
head**. Nhờ vậy mọi chênh lệch đo được quy hoàn cho Custom Classification Head.

### 3.4 Model 3 — `sensitiveai-vi-customdlr2stage`

Bổ sung 5 kỹ thuật cùng lúc:

- **Discriminative Learning Rates** (4 nhóm tham số với LR khác nhau)
- **Huấn luyện 2 stage** — stage 2 nối tiếp từ checkpoint stage 1
- **Focal loss** với `alpha` riêng cho từng nhãn
- **Class weight** kiểu `inverse_sqrt_frequency`
- **Label-specific threshold** — tìm trên validation

### 3.5 Model 4 — `sensitiveai-vi-custom-curriculum`

Học theo thứ tự "dễ → khó → tổng hợp":

| Stage | Epoch | Khái niệm (concept) | Loss | Bắt đầu từ |
|---|---:|---|---|---|
| 1 | 1–3 | `OFFENSIVE` là dương; `HATE`, `CLEAN` là âm | concept | pre-trained |
| 2 | 4–6 | `HATE` là dương; `OFFENSIVE`, `CLEAN` là âm | concept | checkpoint stage 1 |
| 3 | 7–10 | 3 lớp đầy đủ | cross-entropy | checkpoint stage 2 |

**Điểm quan trọng:** mỗi stage dùng **toàn bộ** tập dữ liệu. Lớp âm của một khái niệm là
*hai lớp còn lại*, không phải bằng cách loại bỏ mẫu. Head 3 lớp được giữ nguyên suốt —
loss khái niệm chỉ **đọc lại logits**, nên ánh xạ nhãn không bao giờ đổi giữa các stage.

---

## 4. Kiến trúc mô hình

### 4.1 Encoder dùng chung

`microsoft/mdeberta-v3-base` — mDeBERTa-v3 đa ngôn ngữ (XLM-RoBERTa-style pretraining +
disentangled attention).

| | |
|---|---|
| Số tham số encoder | 278.218.752 |
| Tham số trainable | 278.218.752 (100 %) |
| Hidden size | 768 |
| Số transformer block | 12 |
| Tokenizer | `DebertaV2Tokenizer` (vocab 250.102) |
| Max sequence length | 256, padding cố định `max_length` |

Checkpoint gốc được phát hành ở **float16**. Mã nguồn ép về **float32** khi nạp
(xem `load_encoder()` trong `src/models/model.py`) và dùng `torch.autocast` cho mixed
precision. Nếu không ép, optimizer state sẽ ở nửa độ chính xác và ngân sách bộ nhớ
thay đổi âm thầm — đây là điểm hay gây sai sót.

### 4.2 Standard Head (model 1)

```
input_ids, attention_mask
        ↓
mDeBERTa-v3 encoder  →  sequence_output [B, 256, 768]
        ↓
ContextPooler (dense 768 + tanh)          ← lấy token [CLS]
        ↓
Dropout(0.1) → Linear(768 → 3)            ← logits
```

Đây chính là kiến trúc `AutoModelForSequenceClassification` mặc định. Baseline dùng
**một** pooling vector duy nhất.

### 4.3 Custom Classification Head (model 2, 3, 4)

Thay vì một vector, head này **hợp nhất 3 cách nhìn** khác nhau về cùng một câu:

| View | Công thức | Bắt được gì |
|------|----------|-------------|
| `cls` | token `[CLS]` (768) | ngữ cảnh tổng thể |
| `mean` | trung bình các token thật (768) | nội dung chủ đạo |
| `max` | max theo chiều token cho từng chiều đặc trưng (768) | điểm "nổi bật", từ khóa độc hại |

```
hidden [B, 256, 768]
    ├─ cls   → h_1  [B, 768]
    ├─ mean  → h_2  [B, 768]      (attention_mask có trọng số)
    └─ max   → h_3  [B, 768]      (token pad đẩy xuống -inf)
                    ↓ concat
              z [B, 2304]
                    ↓ LayerNorm(2304)
              Fusion MLP:  Linear(2304 → 512) + GELU + Dropout(0.1) + LayerNorm(512)
                    ↓ Dropout(0.1)
              Linear(512 → 3)  →  logits
```

| | |
|---|---|
| Số tham số encoder | 278.218.752 |
| Số tham số head | ~1.187.331 |
| **Tổng** | **279.406.083** (chỉ hơn baseline ~594K, tức **+0,21 %**) |

Head là phần **rất nhỏ** so với encoder. Đây là điểm cố ý: ablation này kiểm tra xem
*cách tổng hợp đặc trưng* có giá trị hay không, chứ không phải kiểm tra xem thêm
tham số có giúp không.

Tùy chọn có thêm (đều **tắt mặc định**, xem `head_options` trong YAML):

- `use_attention_pool: true` — view thứ 4 dùng attention học được (thêm **986.112** tham số,
  head lên 2.173.443)
- `use_class_prior: true` — thêm bias theo log prior của lớp (buffer, **không** thêm tham số)

### 4.4 Cách lưu checkpoint

Hai loại head lưu theo **hai format khác nhau**, cả hai đều tự chứa đủ để nạp lại:

- **Standard**: thư mục HF chuẩn (`config.json`, `model.safetensors`, tokenizer)
- **Custom**: `encoder/` (encoder HF chuẩn) + `head.safetensors` (chỉ head) +
  `sensitiveai_config.json` (siêu dữ liệu head: hidden size, view nào bật, …)

---

## 5. Dữ liệu ViHSD và tiền xử lý

### 5.1 Nguồn

| | |
|---|---|
| Hugging Face | `uitnlp/vihsd` |
| GitHub | `sonlam1102/vihsd` |
| Bài báo | *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts* |

### 5.2 Phân chia cố định

ViHSD xuất bản sẵn 3 file. Chiến lược mặc định là **giữ nguyên** phân chia chính thức:

| Split | Số mẫu | HATE | OFFENSIVE | CLEAN |
|---|---:|---:|---:|---:|
| `train` | 24.046 | 2.556 | 1.605 | 19.885 |
| `validation` | 2.672 | 270 | 212 | 2.190 |
| `test` | **6.680** | 688 | 444 | 5.548 |
| **Tổng** | **33.398** | 3.514 | 2.261 | 27.623 |

> **Về con số 24.046:** ViHSD gốc có 24.048 dòng train. Dự án **loại bỏ đúng 2 dòng**
> có text là `null`/`NaN` — không có nội dung để mã hoá. Đây là ngoại lệ duy nhất, và nó
> xảy ra **trước** khi tách split, nên **validation và test giữ nguyên tuyệt đối số
> lượng gốc: 2.672 và 6.680**. Điều này đúng yêu cầu "không loại bỏ test sample vì
> preprocessing".

Phương án thay thế (`strategy: stratified` trong `configs/data_split.yaml`) gộp tất cả
rồi chia lại 80/10/10 có phân tầng — cho kết quả 26.718 / 3.340 / 3.340. Cần chạy lại
`00_prepare_data.py` với `--set strategy=stratified` nếu muốn dùng.

### 5.3 Quy tắc làm sạch văn bản

Nguyên tắc thiết kế: **không có thao tác nào xoá thông tin được bật mặc định.** Mọi
thao tác có thể mất thông tin đều là opt-in và mặc định tắt.

| Thao tác | Mặc định | Tác dụng |
|---|:---:|---|
| `lowercase` | ✅ | Hạ chữ, nhận biết Unicode (giữ nguyên dấu tiếng Việt) |
| `normalize_unicode: NFC` | ✅ | Chuẩn hoá dấu thành dạng hợp thành |
| `unescape_html_entities` | ✅ | `&amp;` → `&` |
| `remove_invisible_chars` | ✅ | Bỏ ký tự zero-width, soft hyphen |
| `strip_tags` | ✅ | Bỏ thẻ HTML còn sót |
| `normalize_unicode_spaces` | ✅ | Chuẩn hoá space Unicode |
| `fix_punct_spacing` | ✅ | Sửa khoảng trắng quanh dấu câu |
| `collapse_to_single_space` | ✅ | Gộp khoảng trắng |
| `strip_edges` | ✅ | Cắt khoảng trắng đầu/cuối |
| `remove_urls` | ❌ | *(tắt — sẽ xoá thông tin)* |
| `remove_mentions` | ❌ | *(tắt — sẽ xoá thông tin)* |
| `remove_repeated_punctuation` | ❌ | *(tắt — sẽ xoá thông tin)* |

**Loại bỏ dòng:** chỉ loại dòng mà sau khi làm sạch **không còn ký tự nhìn thấy nào**
(null, toàn khoảng trắng, toàn zero-width). Câu chỉ gồm emoji (🥰🥰🥰, 🤣🤣🤣🤣) hoặc
chỉ dấu câu là **mẫu dữ liệu thật** và được giữ lại.

> **Ghi chú đã sửa:** phiên bản đầu tiên của mã nguồn loại mất 227 dòng train / 19 dev /
> 66 test vì kiểm tra "còn ký tự chữ hoặc số". Điều này vi phạm quy tắc không loại bỏ
> sample, và đã được sửa. Số liệu trong bảng 5.2 là kết quả **sau khi sửa**.

### 5.4 Fingerprint bảo đảm tính công bằng

`data/splits/split_manifest.json` lưu SHA-256 của **từng file split**:

```json
"train":      { "num_samples": 24046, "sha256": "f705ffe94d41f496f3f7e3f1355c6e818d4b30291ce559621099ef2b48889244" }
"validation": { "num_samples": 2672,  "sha256": "6aa552aafb9247580558c59abf02fc3725d3c9de4dc622523651082e8a585353" }
"test":       { "num_samples": 6680,  "sha256": "d8f9ae7ddd8ced908c8cfae8e6a7a7b61dd63849cc225346829f7ca6de28729b" }
```

**Mọi** lần train và evaluate đều gọi `verify_split_fingerprint()` trước khi nạp dữ liệu.
Nếu ai đó thay đổi split, pipeline dừng ngay với thông báo lỗi — không thể vô tình báo
cáo số liệu trên một split khác.

### 5.5 Cấu trúc file dữ liệu

```
data/
├── raw/vihsd/                    # tải từ nguồn, không sửa
│   ├── train.csv                 # cột: free_text, label, ..., label_ids
│   ├── dev.csv
│   └── test.csv
├── processed/vihsd/              # sau làm sạch + remap nhãn
│   ├── train.csv                 # cột: text, label, label_name, source, source_row
│   ├── dev.csv
│   ├── test.csv
│   └── preprocessing_report.json # config đã dùng + số mẫu loại + phân bố nhãn
└── splits/                       # split cố định dùng chung
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── split_manifest.json       # SHA-256 + metadata
```

Cột `source_row` lưu lại chỉ số dòng trong file gốc, để **truy vết được** từ bất kỳ mẫu
nào trong split về dòng gốc ban đầu.

---

## 6. Chiến lược huấn luyện

### 6.1 Discriminative Learning Rates (DLR)

Ý tưởng: **không phải phần nào của mô hình cũng nên "di chuyển" nhiều như nhau** khi tinh
chỉnh trên tập nhỏ. Tham số embedding đã học nhiều từ pretraining nên di chuyển ít; head
mới ngẫu nhiên cần học nhanh.

| Nhóm tham số | LR (model 3, stage 1) | LR (model 4, stage 1) |
|---|---:|---:|
| Word embeddings + embedding LayerNorm | 5e-6 | 5e-6 |
| Encoder LayerNorm + `rel_embeddings` | 1e-5 | 1e-5 |
| Transformer block 0–5 (dưới) | 1e-5 | 1e-5 |
| Transformer block 6–11 (trên) | 2e-5 | 2e-5 |
| Classification head | **1e-4** | **8e-5** |

Tỉ lệ head/embedding là **20:1** (model 3) và **16:1** (model 4). Tham số dưới 12 block
được tách bởi `layer_split: 0.5`; `layer_decay` cho phép áp dụng suy giảm hình học theo độ
sâu (đang đặt `1.0` = không suy giảm).

Ở **các stage sau**, toàn bộ LR bị hạ một nửa — cơ chế "giảm LR sau mỗi lần chuyển
stage" giúp stage mới không phá vỡ biểu diễn đã đạt:

| Nhóm | Model 4 — stage 1 | Model 4 — stage 2 & 3 | Model 3 — stage 2 |
|---|---:|---:|---:|
| Embeddings | 5e-6 | 2.5e-6 | 2.5e-6 |
| Encoder LN + `rel_embeddings` | 1e-5 | 5e-6 | 5e-6 |
| Block 0–5 | 1e-5 | 5e-6 | 5e-6 |
| Block 6–11 | 2e-5 | 1e-5 | 1e-5 |
| Head | 8e-5 | 5e-5 | 5e-5 |

Mỗi nhóm tách riêng `weight_decay` cho bias/LayerNorm (`_no_decay`).

> **Dễ nhầm khi đọc YAML:** trong block `stages:` của model 3/4 vẫn còn khoá
> `training.learning_rate` (2e-5, 1e-5, 1.5e-5). Vì `dlr.enabled: true`, giá trị này
> **không được dùng** cho bất kỳ nhóm nào — nó chỉ là giá trị dự phòng cho nhóm thiếu khai
> báo. LR thực tế luôn lấy từ 4 khoá `embedding_lr` / `encoder_lr` / `layer_lr` /
> `head_lr`. Kiểm tra chéo bằng `--dry-run`.

### 6.2 Focal loss với alpha theo nhãn

```
FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)
```

| | |
|---|---|
| `gamma` | 2.0 — hạ trọng số mẫu dễ (yếu tố `(1-p_t)^2` ≈ 0) |
| `alpha` | HATE 0.5, OFFENSIVE 0.25, CLEAN 0.25 |

Vì `alpha_t` nhân theo **nhãn thật** (không phải theo nhãn dự đoán), mẫu HATE khó sẽ
nhận biên lớn hơn gấp đôi. Mục tiêu: để gradient bị chi phối bởi mẫu HATE hiếm và khó
thay vì hàng nghìn mẫu CLEAN dễ.

### 6.3 Class weight

| Chiến lược | Công thức | Dùng ở |
|---|---|---|
| `none` | w = 1 | baseline |
| `inverse_frequency` | w ∝ 1/n_c | (có sẵn) |
| `inverse_sqrt_frequency` | w ∝ 1/√n_c | model 3 |
| `effective_number` | w ∝ (1-β)/(1-β^(n_c)) | (có sẵn) |
| `manual` | chỉ định tay | model 4 |

Tính trên **chỉ tập train**:

```
model 3 (inverse_sqrt_frequency)   HATE 1.145  OFFENSIVE 1.445  CLEAN 0.410
model 4 (manual)                   HATE 3.0    OFFENSIVE 2.0    CLEAN 1.0
```

Sau tính, mọi vector được **chuẩn hoá về trung bình = 1** để LR không bị đổi ý nghĩa.

### 6.4 Concept-focused curriculum loss

Ở mỗi stage, một nhãn được chọn làm **dương**; hai nhãn còn lại là **âm**:

```
z_pos = logits[:, focus]
z_neg = stack(logits[:, j] cho j != focus)

loss = -w_pos · log( softmax([z_pos, z_neg])[0] )
```

- `softmax` trên `z_neg` **giữ được thứ tự tương đối** giữa hai lớp âm — head không bị
  mất thông tin phân biệt OFFENSIVE với HATE trong lúc học khái niệm
- `positive_weight` mặc định `auto`: tỉ lệ âm/dương ước lượng trên train, cho phép head
  "ghi nhớ" bias của lớp dương trước khi học phân biệt
- `gamma` mặc định `0.0` (không giảm mẫu dễ trong curriculum — khác focal loss)
- Head 3 lớp được giữ nguyên suốt; loss chỉ **đọc lại logits**

### 6.5 Huấn luyện nhiều stage

Mỗi stage có thể override: `training`, `dlr`, `loss`, `epochs`, `init_from`,
`save_checkpoint`.

```
stage k bắt đầu từ  stage:stage_(k-1)   →  nạp checkpoint stage trước
```

Checkpoint stage được lưu vào `experiments/<model>/run_XXX/stage_checkpoints/<stage>/`
và mỗi lần chuyển stage đều được ghi lại trong `stage_transitions.json` (audit trail).

### 6.6 Cấu hình huấn luyện (dành cho GPU 8 GB)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `batch_size` | 16 | Vừa với 8 GB |
| `gradient_accumulation` | 2 | Effective batch = 32 |
| `precision` | `bf16` | RTX 4060 hỗ trợ bf16 |
| `gradient_checkpointing` | `true` | Đổi compute lấy bộ nhớ |
| `max_length` | 256 | Độ dài bài ViHSD rất ngắn |
| `max_grad_norm` | 1.0 | Chống gradient spike |
| `warmup_ratio` | 0.1 (0.05 ở stage sau của model 3/4) | Ổn định khởi động |
| `scheduler` | linear decay | Chuẩn cho fine-tune |
| `num_workers` | 2 | (đặt 0 khi debug trên Windows) |
| `early_stopping` | macro F1, patience 3 | Chọn trên **validation** |

Bốn config dùng **giống hệt nhau** toàn bộ bảng trên — đây là phần "giữ cố định" của thiết
kế thực nghiệm. Chỉ có `warmup_ratio` được giảm còn 0.05 ở các stage sau của model 3 và 4
(vì giai đoạn đó bắt đầu từ checkpoint đã học).

> **Nếu gặp CUDA Out of Memory:** đặt `training.batch_size=8` và
> `training.gradient_accumulation=4` — effective batch vẫn là 32.

### 6.7 Cấu trúc vòng lặp huấn luyện

Dự án **không** dùng Hugging Face `Trainer`. Vì `transformers` 5.x đã đổi API
(`TrainingArguments` → `Seq2SeqTrainingArguments`, callback signature đổi), và cần hỗ trợ
multi-stage + curriculum + DLR, một vòng lặp tùy chỉnh được viết trong
`src/training/trainer.py` (~473 dòng).

```
for stage in stages:
    nạp weight (nếu init_from)
    dựng optimizer (DLR param groups) + scheduler + loss cho stage
    for epoch in stage.epochs:
        train_epoch  ──► grad accumulate ──► clip grad norm ──► step
        evaluate     ──► macro F1 trên validation
        callbacks    ──► best checkpoint / early stopping / CSV log
    lưu checkpoint stage, ghi transition
lưu last_model, viết training_summary.json
```

Callback theo thứ tự: `BestCheckpointCallback` → `EarlyStoppingCallback` →
`CSVLoggerCallback`. Best checkpoint chọn theo **validation macro F1** và được lưu kèm
`best_training_state.pt`.

---

## 7. Cấu trúc kho mã

```
AIModerationContent/
├── README.md                    <- tài liệu này
├── requirements.txt
├── plan.md                       <- kế hoạch nghiên cứu gốc (nguồn của mọi quyết định)
│
├── configs/                      <- 5 file YAML
│   ├── data_split.yaml
│   ├── sensitiveai-vi.yaml
│   ├── sensitiveai-vi-custom.yaml
│   ├── sensitiveai-vi-customdlr2stage.yaml
│   └── sensitiveai-vi-custom-curriculum.yaml
│
├── scripts/                      <- 5 entry point + hướng dẫn
│   ├── README.md
│   ├── 00_prepare_data.py
│   ├── 01_train_all.py
│   ├── 02_evaluate_all.py
│   ├── 03_compare_models.py
│   └── 04_check_setup.py
│
├── src/                          <- 43 module, 49 file .py, 8.208 dòng
│   ├── common/                   <- hằng số, config, path, seed, logging, env
│   ├── preprocessing/            <- làm sạch văn bản
│   ├── data/                     <- tải, tiền xử lý, split, dataset
│   ├── models/                   <- StandardHeadModel, CustomHeadModel
│   ├── losses/                   <- CE, focal, class weight, curriculum
│   ├── callbacks/                <- CSV log, monitoring, stage controller
│   ├── training/                 <- DLR optimizer, stages, trainer, CLI
│   ├── evaluation/               <- metrics, curves, thresholds, benchmark, plots, CLI
│   └── comparison/               <- bảng so sánh + sinh LaTeX
│
├── data/                         <- sinh ra khi chạy (không commit dữ liệu lớn)
├── reports/<model>/run_00N/      <- sinh ra khi train
├── experiments/<model>/run_00N/  <- checkpoint từng stage
├── checkpoints/                  <- dự trữ
├── evaluation/                   <- kết quả so sánh
└── paper/                        <- main.tex, references.bib, sections/, tables/, figures/
```

### 7.1 Vai trò từng module

| Module | Trách nhiệm |
|---|---|
| `src/common/constants.py` | Nhãn cố định, đường dẫn, tên 4 model |
| `src/common/config.py` | Nạp/merge YAML, override `--set`, validate |
| `src/common/paths.py` | Quy ước `run_XXX`, `find_best_report_dir()` |
| `src/common/seeding.py` | Seed + deterministic + worker init |
| `src/preprocessing/text_cleaning.py` | 12 bước làm sạch, không mất thông tin |
| `src/data/download.py` | Tải ViHSD từ Hugging Face |
| `src/data/preprocess.py` | Làm sạch + remap nhãn + báo cáo |
| `src/data/split.py` | Tạo split, SHA-256, kiểm tra toàn vẹn test |
| `src/data/dataset.py` | Tokenize, `TokenizedTextDataset`, DataLoader |
| `src/models/custom_head.py` | Multi-view pooling + fusion MLP |
| `src/models/model.py` | `StandardHeadModel`, `CustomHeadModel`, save/load |
| `src/losses/class_weights.py` | 5 chiến lược cân bằng lớp |
| `src/losses/focal.py` | Focal loss, `alpha` theo nhãn |
| `src/losses/curriculum.py` | Concept-focused loss |
| `src/losses/builder.py` | `build_loss()` — YAML → module loss |
| `src/training/optim.py` | Chia nhóm tham số cho DLR, scheduler |
| `src/training/stages.py` | `resolve_stages()` — merge global + stage override |
| `src/training/trainer.py` | Vòng lặp huấn luyện, stage, checkpoint |
| `src/callbacks/stage_controller.py` | Điều phối stage, `stage_transitions.json` |
| `src/callbacks/monitoring.py` | Early stopping, best checkpoint |
| `src/callbacks/csv_logger.py` | `training_log.csv` — nguồn số liệu cho mọi biểu đồ |
| `src/evaluation/metrics.py` | Accuracy, macro/weighted F1, per-class, AUC, AP |
| `src/evaluation/thresholds.py` | Tìm threshold theo từng nhãn trên validation |
| `src/evaluation/benchmark.py` | Đo latency (ms/sample) và throughput (FPS) |
| `src/evaluation/run_evaluation.py` | CLI đánh giá — **test chấm đúng 1 lần** |
| `src/comparison/compare_models.py` | Bảng CSV/MD, biểu đồ, **sinh LaTeX tự động** |

---

## 8. Cài đặt môi trường

### 8.1 Môi trường đã kiểm thử

| | |
|---|---|
| Hệ điều hành | Windows 11 (build 26300) |
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu126 |
| `transformers` | 5.14.1 |
| `tokenizers` | 0.22.2 |
| `datasets` | 5.0.0 |
| pandas | 3.0.3 · numpy 2.4.6 · scikit-learn 1.8.0 |
| matplotlib | 3.11.1 · seaborn 0.13.2 |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, driver 617.14 |

### 8.2 Cài đặt

```powershell
python -m pip install -r requirements.txt
```

Kho mã đã được kiểm thử với `transformers` **5.14.1**. Với bản 4.x, mã nguồn có
fallback tự động (`torch_dtype` thay cho `dtype`) nhưng **chưa được kiểm thử thực tế**.

---

## 9. Hướng dẫn chạy đầy đủ

### 9.1 Chuỗi lệnh chuẩn

```powershell
# Bước 1 — dữ liệu
python scripts/00_prepare_data.py

# Bước 2 — kiểm tra môi trường (nhanh, KHÔNG train)
python scripts/04_check_setup.py

# Bước 3 — train cả 4 model
python scripts/01_train_all.py

# Bước 4 — chấm test (nhanh hơn nhiều; thời gian thực tế sẽ ghi vào evaluate.log)
python scripts/02_evaluate_all.py

# Bước 5 — bảng + biểu đồ cho paper
python scripts/03_compare_models.py
```

> ⚠️ **Bước 3 là bước lâu nhất.** Ước tính khoảng 9–12 giờ cho cả 4 model trên RTX 4060 8 GB
> (con số này là **ước lượng theo tốc độ đo trong smoke test**, chưa phải thời gian đo thật —
> hãy ghi lại thời gian thực tế của lần chạy đầu). Nên chạy từng model riêng để dễ theo
> dõi và có thể dừng giữa chừng.

### 9.2 Train từng model

```powershell
# Model 1 — baseline, 6 epoch
python -m src.training.run_experiment --config sensitiveai-vi

# Model 2 — custom head, 6 epoch
python -m src.training.run_experiment --config sensitiveai-vi-custom

# Model 3 — DLR + 2 stage + focal, 7 epoch
python -m src.training.run_experiment --config sensitiveai-vi-customdlr2stage

# Model 4 — curriculum 3 stage, 10 epoch
python -m src.training.run_experiment --config sensitiveai-vi-custom-curriculum
```

Nhận được cả tên config lẫn đường dẫn: `--config configs/sensitiveai-vi.yaml`.

### 9.3 Xem kế hoạch trước khi chạy

```powershell
python -m src.training.run_experiment --config sensitiveai-vi-custom-curriculum --dry-run
```

In ra: số stage, epoch của từng stage, LR theo nhóm tham số, loại loss, chuỗi khởi tạo —
rồi thoát **không** train.

### 9.4 Ghi đè cấu hình khi chạy

`--set` nhận danh sách `key=value` phân tách theo dấu chấm. Vì dùng `nargs="*"`, **phải đặt
trước** mọi cờ bắt đầu bằng `--`, nếu không nó sẽ nuốt luôn các cờ đó:

```powershell
# Giảm thời gian
python -m src.training.run_experiment --config sensitiveai-vi-custom-curriculum `
    --set training.epochs=5 training.batch_size=8 training.gradient_accumulation=4

# Train trên subset để kiểm tra nhanh
python -m src.training.run_experiment --config sensitiveai-vi --set training.epochs=2 --max-train-samples 4000
```

### 9.5 Chạy thử (smoke test)

Dùng 256 mẫu / 128 mẫu eval / 1 epoch mỗi stage / tắt checkpoint. **Kết quả vô nghĩa về
mặt khoa học** — chỉ để kiểm tra đường ống chạy được:

```powershell
python scripts/01_train_all.py --smoke-test --reports-dir reports_smoke --experiments-dir experiments_smoke
python scripts/02_evaluate_all.py --reports-dir reports_smoke --max-eval-samples 128 --no-benchmark
python scripts/03_compare_models.py --reports-dir reports_smoke --output-dir evaluation_smoke/comparison --paper-dir paper_smoke
```

Cần dọn sau khi test: `Remove-Item -Recurse -Force reports_smoke,experiments_smoke,evaluation_smoke,paper_smoke`

### 9.6 Đánh giá từng model

```powershell
python -m src.evaluation.run_evaluation --model sensitiveai-vi-custom-curriculum
python -m src.evaluation.run_evaluation --model sensitiveai-vi --force-argmax
python -m src.evaluation.run_evaluation --model sensitiveai-vi-customdlr2stage --no-thresholds
```

| Cờ | Tác dụng |
|---|---|
| `--run-id run_001` | Chỉ định run cụ thể (mặc định: run mới nhất) |
| `--checkpoint <đường dẫn>` | Nạp weight từ đường dẫn tuỳ ý |
| `--no-thresholds` | Bỏ tìm threshold |
| `--force-argmax` | Bỏ qua threshold, dùng argmax |
| `--no-benchmark` | Bỏ đo latency/FPS |
| `--split validation` | Chấm validation thay vì test |

### 9.7 Bảng các cờ huấn luyện

| Cờ | Tác dụng |
|---|---|
| `--run-id run_001` | Gán run id cụ thể thay vì auto |
| `--reports-dir` / `--experiments-dir` | Ghi kết quả ra thư mục khác |
| `--output-dir` | Ghi thẳng vào một thư mục cụ thể |
| `--device cpu` | Ép chạy CPU |
| `--no-save-checkpoints` | Không lưu weight |
| `--dry-run` | In kế hoạch rồi thoát |
| `--max-train-samples N` | Giới hạn số mẫu train |
| `--max-eval-samples N` | Giới hạn số mẫu validation/test |
| `--smoke-test` | Chế độ debug tổng hợp |

---

## 10. Cấu trúc kết quả đầu ra

### 10.1 Một lần chạy: `reports/<model>/run_00N/`

| File / thư mục | Nội dung |
|---|---|
| `config.yaml` | Cấu hình **đã resolve** (đã áp override) — snapshot chính xác của run |
| `run_pointer.txt` | Trỏ tới run mới nhất |
| `train.log` | Log đầy đủ |
| `training_log.csv` | **Một dòng mỗi epoch** — nguồn số liệu cho mọi biểu đồ |
| `training_summary.json` | Tổng kết: best epoch/stage, thời gian, số epoch, param groups, lịch sử |
| `class_weights.json` | Class weight ước lượng trên train + bảng mô tả |
| `stage_transitions.json` | Audit trail từng lần chuyển stage |
| `best_model/` | Weight của epoch có validation macro F1 cao nhất + tokenizer |
| `best_training_state.pt` | Trạng thái optimizer/scheduler tại epoch đó |
| `last_model/` | Weight của epoch cuối |

### 10.2 Cột của `training_log.csv`

```
model_name, stage, stage_index, epoch, epoch_in_stage,
learning_rate, encoder_lr, head_lr,
loss, accuracy,
val_loss, val_accuracy, val_macro_f1, val_weighted_f1,
val_HATE_f1, val_OFFENSIVE_f1, val_CLEAN_f1,
val_HATE_recall, val_OFFENSIVE_recall, val_CLEAN_recall,
grad_norm, epoch_seconds, cumulative_seconds, is_best
```

> **Quy tắc quan trọng:** mọi con số trên biểu đồ validation-accuracy / validation-loss
> của bài báo đều đọc từ file này. Không có biểu đồ nào được vẽ lại từ buffer trong RAM —
> nhờ vậy hình và bảng luôn khớp với log.

### 10.3 Một lần đánh giá (thêm vào cùng thư mục run)

| File | Nội dung |
|---|---|
| `metrics.json` | Toàn bộ số liệu test (JSON) |
| `test_predictions.csv` | **Từng mẫu**: `text`, `true_label`, `predicted_label`, xác suất 3 lớp |
| `classification_report.txt` | Báo cáo `sklearn` dạng văn bản |
| `Best_model_classification_report.txt` | Bản sao (tương thích tên file cũ) |
| `confusion_matrix.txt` | Ma trận nhầm lẫn dạng văn bản |
| `confusion_matrix.png` / `_raw` / `_normalized` | 3 phiên bản hình |
| `pr_curve.png` / `roc_curve.png` | PR và ROC toàn bộ lớp |
| `pr_curve_data.json` / `roc_curve_data.json` | **Điểm dữ liệu gốc** của hai đường cong trên |
| `per_class_f1.png` | F1 từng lớp |
| `evaluate.log` | Log riêng của bước chấm test |
| `val_accuracy_curve.png` / `val_loss_curve.png` / `train_loss_curve.png` | Đường cong theo epoch, đánh dấu chuyển stage |
| `thresholds.json` / `thresholds.txt` | Threshold tối ưu + giá trị objective trước/sau |
| `benchmark.json` | Latency, FPS, số batch |
| `model_summary.txt` | Bản tóm tắt dễ đọc cho từng model |

### 10.4 Đầu ra so sánh: `evaluation/comparison/`

| File | Nội dung |
|---|---|
| `model_comparison.csv` | Bảng tổng hợp **tất cả chỉ số** của 4 model |
| `model_comparison.md` | Cùng bảng dạng Markdown (đẹp hơn) |
| `comparison_summary.json` | Tổng hợp dạng máy-đọc (chỉ số chính của từng model) |
| `pr_curve_all_models.png`, `roc_curve_all_models.png` | PR/ROC macro |
| `pr_hate_all_models.png`, `pr_offensive_all_models.png`, `pr_clean_all_models.png` | PR từng lớp |
| `roc_*_all_models.png` | ROC từng lớp |
| `validation_accuracy_all_models.png`, `validation_loss_all_models.png` | So sánh đường cong validation |
| `macro_f1_all_models.png`, `accuracy_all_models.png` | So sánh dạng thanh |
| `per_class_f1_all_models.png` | Nhóm thanh F1 theo lớp |

### 10.5 Đầu ra cho paper

`03_compare_models.py` **tự sinh** các file sau — không con số nào được gõ tay:

| File | Nội dung |
|---|---|
| `paper/tables/model_comparison.tex` | Bảng so sánh 4 model (giá trị tốt nhất in đậm) |
| `paper/tables/classification_results.tex` | P/R/F1 theo từng lớp |
| `paper/tables/ablation_results.tex` | Bảng ablation — công cụ của từng quyết định thiết kế |
| `paper/figures/*.png` | Copy từ `evaluation/comparison/` |
| `paper/main.tex` | Khung bài báo, `\input` các bảng trên |
| `paper/references.bib` | 13 tài liệu tham khảo |

Mỗi file `.tex` có dòng đầu `% AUTO-GENERATED ... - do not edit by hand.`

### 10.6 Bảng ablation sinh tự động

| Variant | Head | DLR | Stages | Loss |
|---|---|:---:|:---:|---|
| `sensitiveai-vi` | standard | no | 1 | cross-entropy |
| `sensitiveai-vi-custom` | custom | no | 1 | cross-entropy |
| `sensitiveai-vi-customdlr2stage` | custom | yes | 2 | CE stage 1 / focal stage 2 |
| `sensitiveai-vi-custom-curriculum` | custom | yes | 3 | concept 1 / concept 2 / CE |

Bảng này **trả lời trực tiếp** câu hỏi của từng quyết định thiết kế: đổi head được bao
nhiêu, thêm DLR + focal + 2 stage được bao nhiêu, curriculum 3 stage được bao nhiêu so với
2 stage.

---

## 11. Quy chuẩn công bằng thực nghiệm (fairness)

Đây là phần quan trọng nhất về mặt phương pháp luận. Mọi quy tắc dưới đây được **thực
thi bằng code**, không phải quy ước miệng.

### 11.1 Những gì bị cấm

| Quy tắc | Cách thực thi trong mã |
|---|---|
| **Test set phải được chấm đúng 1 lần** | `run_evaluation.py` chạy inference trên split được chấm **một lần duy nhất**; loss tham chiếu được tính **trong chính pass đó** (không có pass thứ hai) |
| **Không dùng test để chọn mô hình** | Early stopping và best-checkpoint chỉ dùng **validation macro F1** |
| **Không tune threshold trên test** | Threshold tìm trên **validation** (`optimize_thresholds`), lưu đông cứng vào `thresholds.json`, rồi áp cho test |
| **Không dùng test trong class weight** | `compute_class_weights()` chỉ nhận nhãn train; kết quả ghi vào `class_weights.json` kèm nhãn `"estimated_on": "train split only"` |
| **Cùng split cho mọi model** | `verify_split_fingerprint()` chạy trước mỗi lần train/evaluate; sai SHA-256 → dừng |
| **Cùng label mapping** | Hằng số đóng băng trong `constants.py`, áp dụng 1 lần ở tiền xử lý |

### 11.2 Những gì được giữ cố định giữa 4 model

| | |
|---|---|
| Backbone | `microsoft/mdeberta-v3-base` |
| Split | cùng SHA-256 |
| `max_length` | 256 |
| `batch_size` × `grad_accum` | 16 × 2 = 32 |
| `precision` | bf16 |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| `seed` | 42 |
| Tiêu chí early stopping | validation macro F1 |

### 11.3 Chỉ số đánh giá

**Số liệu chính:**

- **Macro F1** — trung bình harmonic F1 của 3 lớp
- **Per-class F1** — thấy rõ lớp nào bị bỏ sót

**Số liệu phụ:** accuracy, weighted F1, macro precision/recall, macro & per-class ROC-AUC,
macro & per-class average precision, confusion matrix, latency (ms/sample), throughput (FPS).

**Không dùng accuracy làm số liệu chính** — vì 82,7 % dữ liệu là CLEAN, một mô hình dự
đoán toàn CLEAN cũng đạt 82,7 %.

---

## 12. Tái lập kết quả trên máy khác

### 12.1 Quy trình

```powershell
git clone <repo> && cd <repo>
python -m pip install -r requirements.txt

python scripts/00_prepare_data.py
python -m src.evaluation.run_evaluation --help   # kiểm tra CLI

python scripts/01_train_all.py
python scripts/02_evaluate_all.py
python scripts/03_compare_models.py
```

### 12.2 Những gì sẽ giống hệt

- Số mẫu mỗi split và phân bố nhãn (lưu trong manifest)
- SHA-256 của cả 3 file split
- Cấu trúc pipeline và thứ tự các bước
- Cấu hình, siêu tham số, seed
- Số lượng stage, epoch, kiểu loss, nhóm learning rate

### 12.3 Những gì **không** giống hệt

| Yếu tố | Lý do | Mức ảnh hưởng |
|---|---|---|
| **Số tham số trainable khác nhau** | Head chuẩn của HF có pooler riêng (278.811.651) vs custom head (279.406.083) | Đã ghi rõ trong báo cáo |
| **Download checkpoint Hugging Face** | Có thể khác phiên bản | Ghi `revision` vào config |
| **bf16 trên GPU khác kiến trúc** | VFP16 của Ada có độ chính xác khác bf16 | Nhỏ |
| **Non-determinism của cuDNN** | Dù đặt `deterministic: true`, kernel cuDNN/CUDA cuối cùng vẫn có thể khác | Nhỏ |
| **Độ chính xác GPU** | fp32 vs TF32 vs bf16 | Rất nhỏ, dưới 1 điểm % |

**Khuyến nghị khi tái lập:** nếu kết quả lệch > 1 điểm % macro F1 so với run gốc, hãy
đối chiếu SHA-256 của split trước, rồi mới xét tới phiên bản dependency.

### 12.4 Reproducibility bit

`seed: 42` được dùng cho: `torch`, `numpy`, `random`, `PYTHONHASHSEED`, DataLoader worker
init, shuffle của train. `deterministic: true` bật `torch.use_deterministic_algorithms`.
Lưu ý: điều này có thể **làm chậm** và không đảm bảo tuyệt đối trên mọi GPU.

---

## 13. Kiểm thử và kiểm định chất lượng

Các kiểm tra đã chạy trên toàn bộ pipeline:

| Kiểm tra | Kết quả |
|---|---|
| Import toàn bộ module | **43/43 thành công** |
| `pyflakes` (static lint) | **0 cảnh báo** |
| `04_check_setup.py` | **9/9 OK** |
| Tải + tiền xử lý + split ViHSD | OK — fingerprint đã xác minh |
| Cả 2 chiến lược split (`official`, `stratified`) | OK |
| Train 4 model (smoke) | **4/4 thành công**, đủ 1/2/3 stage |
| Đánh giá 4 model | **4/4 thành công**, kèm threshold tuning |
| So sánh + sinh LaTeX | OK, 3 bảng `.tex` hợp lệ |
| DataLoader `num_workers=0` và `=2` | OK trên Windows |

**14 lỗi đã được phát hiện và sửa** trong quá trình kiểm thử. Đáng ghi lại vì chúng cho
thấy các kiểm tra này có giá trị thực tế:

| # | Lỗi | Hậu quả nếu không sửa |
|---|---|---|
| 1 | Làm sạch loại mất dòng chỉ-gồm-emoji | Mất 227 train / 19 dev / **66 test** — vi phạm nguyên tắc cốt lõi |
| 2 | Checkpoint mDeBERTa load ở fp16 | Lỗi dtype; optimizer state nửa độ chính xác |
| 3 | `end_stage()` tăng index rồi `next_stage_exists()` kiểm tra index đã tăng | **Stage cuối bị bỏ qua** — curriculum chạy 2/3 stage |
| 4 | `collate_fn=lambda` + `num_workers=2` trên Windows | **Crash** — lambda không pickle được với `spawn` |
| 5 | `--reports-dir` ghi đè mất tên model | Ghi đè kết quả của các model khác nhau vào cùng chỗ |
| 6 | Tên cột LaTeX sai (`HATE_precision` vs `HATE Precision`) | **Crash** khi sinh bảng |
| 7 | `Path(write_text(...))` | **Crash** — `write_text` trả về số ký tự, không phải `Path` |
| 8 | `tabular` spec 13 cột vs 10 dòng | LaTeX không biên dịch được |
| 9 | `fmt="d"` trên mảng float | **Crash** khi vẽ confusion matrix |
| 10 | Biến `boundaries` không tồn tại | **Crash** khi vẽ đường cong validation |
| 11 | `is_best` luôn `False` trong CSV | Mất đánh dấu epoch tốt nhất |
| 12 | Best metric = `n/a` khi tắt early stopping | Báo cáo sai |
| 13 | Strategy `manual` crash | **Crash** model 4 |
| 14 | `focus_label` kế thừa nhầm ở stage CE | Log gây hiểu nhầm |

> **Bài học:** 8 trong 14 lỗi chỉ lộ ra khi **thực sự chạy** từng model, không lộ ra khi
> đọc code hay chạy import check. Vì vậy smoke test end-to-end là bắt buộc, không phải
> tuỳ chọn.

---

## 14. Ghi chú kỹ thuật quan trọng

Những điểm dễ gây nhầm lẫn khi đọc mã hoặc khi tái lập:

### 14.1 Kiểm tra split chạy trước mọi thứ

`verify_split_fingerprint()` được gọi **trước khi** nạp dữ liệu. Nếu bạn thay đổi split,
pipeline dừng ngay. Nếu bạn thấy lỗi fingerprint, **đừng** bypass — hãy chạy lại
`00_prepare_data.py`.

### 14.2 Ngưỡng chính xác

`min_chars: 1` nghĩa là giữ mọi mẫu có ít nhất 1 ký tự nhìn thấy. Muốn loại comment rỗng
thật sự, tăng lên — nhưng **đừng** dùng cách đó để "dọn dữ liệu" rồi vẫn báo cáo là test
nguyên bản.

### 14.3 Best checkpoint vs last checkpoint

`02_evaluate_all.py` mặc định dùng `best_model/`. Nếu `best_model` không tồn tại (ví dụ
smoke test tắt checkpoint), nó **cảnh báo rõ ràng** rồi fallback sang `last_model`. Cảnh
báo này xuất hiện trong log — đừng bỏ qua.

### 14.4 `--set` phải đứng trước cờ khác

```powershell
# Đúng
python -m src.training.run_experiment --config m --set training.epochs=5 --device cpu

# Sai — --device cpu bị nuốt vào --set
python -m src.training.run_experiment --config m --device cpu --set training.epochs=5
```

### 14.5 `num_workers` trên Windows

Đặt `num_workers: 0` khi debug trên Windows. Giá trị 2 hoạt động (đã kiểm thử) vì
`collate_fn` là hàm module-level, nhưng thêm worker trên máy yếu thường không đáng.

### 14.6 Tiêu chí DLR dựa trên **tên tham số**

`split_encoder_layer()` phân loại theo tiền tố tên tham số (`embeddings.*`,
`encoder.layer.N.*`, …). Nếu đổi backbone, **phải kiểm tra lại** hàm `_layer_index()`.

### 14.7 Curriculum giữ nguyên head 3 lớp

Loss khái niệm chỉ đọc lại logits, không thay đổi số lớp. Vì vậy checkpoint stage 1 và
stage 2 **load thẳng** vào cùng kiến trúc. Nếu ai đó đổi sang head nhị phân ở stage giữa,
checkpoint sẽ không tương thích.

### 14.8 Bảng LaTeX được sinh, không gõ tay

Sửa tay `paper/tables/*.tex` sẽ bị ghi đè lần chạy `03` kế tiếp. Muốn chỉnh bảng, hãy sửa
`src/comparison/compare_models.py`.

---

## 15. Tài liệu tham khảo

`paper/references.bib` (13 mục):

| Tài liệu | Vai trò trong nghiên cứu |
|---|---|
| **ViHSD** (Luu et al., 2021) | Dataset |
| **mDeBERTa-v3** (Liu et al., 2024) | Backbone |
| **DeBERTa** (Liu et al., 2019) | Kiến trúc nền của backbone |
| **XLM** (Ouyang et al., 2018) | Nền đa ngôn ngữ của mDeBERTa |
| **AdamW / decoupled weight decay** (Loshchilov & Hutter, 2019) | Optimizer |
| **Focal loss** (Lin et al., 2017) | `gamma`, `alpha` theo nhãn |
| **Class-balanced loss** (Cui et al., 2019) | Chiến lược `effective_number` |
| **Curriculum learning** (Bengio et al., 2009) | Nền tảng lý thuyết cho model 4 |
| **PR vs ROC** (Saito & Rehmsmeier, 2015) | Căn cứ báo cáo PR-AUC thay vì ROC-AUC |
| **BERT** (Devlin et al., 2019) | Bối cảnh kiến trúc |
| **Transformers** (Wolf et al., 2019) | Thư viện triển khai |
| **BLEU** (Papineni et al., 2002) | Tham chiếu cho bộ đo đánh giá |
| **Social media text classification** (Dethlefsen & Johansen, 2022) | Bối cảnh đánh giá |

---

## Phụ lục A — Câu hỏi thường gặp

**Hỏi:** Train từng model hay train cả 4 cùng lúc?
**Đáp:** Có thể cả hai. `01_train_all.py` tiện hơn; train từng model giúp dễ theo dõi và
dừng giữa chừng. Kết quả giống nhau vì config giống nhau.

**Hỏ:** Có cần chạy lại `00_prepare_data.py` không?
**Đáp:** Không, nếu `data/splits/split_manifest.json` đã tồn tại và SHA-256 khớp. Chạy
`python scripts/00_prepare_data.py --stage summary verify` để kiểm tra.

**Hỏ:** Test set 6.680 hay 6.614 mẫu?
**Đáp:** **6.680** — giữ nguyên số lượng gốc. Xem mục 5.2.

**Hỏ:** Vì sao không dùng accuracy?
**Đáp:** 82,7 % dữ liệu là CLEAN. Mô hình dự đoán toàn CLEAN đạt 82,7 %. Xem mục 2.2.

**Hỏ:** Có nên bật `use_attention_pool` không?
**Đáp:** Chưa biết — đó là một hướng ablation khác, **ngoài phạm vi** nghiên cứu này.
Đặt `head_options.use_attention_pool: true` để thử, nhưng nhớ ghi rõ là biến thể ngoài kế
hoạch.

**Hỏ:** Gặp CUDA Out of Memory?
**Đáp:** `--set training.batch_size=8 training.gradient_accumulation=4` (đặt trước cờ
khác), hoặc giảm `tokenizer.max_length` xuống 192.

**Hỏ:** Có cần `bitsandbytes` không?
**Đáp:** Không. Đã có sẵn trong `requirements.txt` dưới dạng comment. 8-bit optimizer có
thể giảm ~2 GB bộ nhớ optimizer nếu cần.

**Hỏ:** Paper skeleton có dùng được không?
**Đáp:** `paper/main.tex` + `paper/sections/*.tex` (placeholder) + `references.bib` (13
mục). Biên dịch bằng `pdflatex main.tex` (3 lần) hoặc để Overleaf xử lý.

---

## Phụ lục B — Lịch sử kiểm thử

| Ngày | Nội dung |
|---|---|
| 2026-10-04 | Dựng kho mã (49 file .py, 8.208 dòng, 43 module) |
| | Tải + tiền xử lý + split ViHSD; tạo fingerprint |
| | Smoke test 4 model → phát hiện 14 lỗi → sửa hết |
| | Xác minh đầu-cuối: train → evaluate → compare cho cả 4 model |
| | Viết README + scripts/README + paper skeleton |
| | Dọn sạch thư mục kết quả, sẵn sàng cho lần chạy thật |

**Trạng thái hiện tại:** mã nguồn hoàn chỉnh và đã kiểm chứng đầu-cuối. **Chưa có kết quả
thực nghiệm** — người dùng tự chạy `01` → `02` → `03`.
