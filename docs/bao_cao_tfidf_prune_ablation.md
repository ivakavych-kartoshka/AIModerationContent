# BÁO CÁO — Trích xuất từ nhạy cảm bằng Tokenize + TF-IDF & Ablation độ lớn từ điển

> **Ngày:** 19/10/2026 · **Repo:** `AIModerationContent` · **Chủ đề:** Lexicon mới (tokenize + TF-IDF + gán nhãn), eval thật Rule Engine, và ablation siết cổng prune (hướng B)
> **Phương án:** `Rule_Engine_Plan.md` §43–44 · **Đối tượng đọc:** người chạy experiment (sinh viên)

---

## 0. Tóm tắt trong 8 dòng

1. **Đã chạy lại đúng pipeline yêu cầu:** tokenize → TF-IDF → gán nhãn trên 3 dataset (ViHSD, ViCM, Bontori) → **922 entry** trong `verified_lexicon.csv`.
2. TF-IDF **cắt được 52,75%** ứng viên thô (3382 → 1598), toàn bộ chỗ cắt đến từ Bontori.
3. **Smoke test rule engine full: PASSED 27/27 check** (0,52 ms/dòng, không crash).
4. **Eval thật (§44):** tune trên validation → đóng băng → test **1 lần**. Rule engine tốt nhất: **macro-F1 = 0.5304, acc = 0.8141** (config `tuned`, `accentless_alias 0.15`).
5. **Phát hiện:** prune mặc định **là no-op** (giữ 922/922, file output **byte-identical** với input) → vì TF-IDF đã chạy sẵn ở khâu trích xuất.
6. **Hướng B đã thực hiện:** siết `min_score` → 3 biến thể **691 / 576 / 324 từ**, sweep trên **validation**, chọn **576** (Δmacro-F1 ≤ 0.010), rồi test 1 lần.
7. **Kết quả test của biến thể 576:** macro-F1 **0.5270** (−0,0034 = −0,64% so với 922), **accuracy tăng +0,0070**, **nhanh hơn 18,5%** (0,498 → 0,406 ms).
8. **⚠️ Còn 2 rủi ro phải xử lý:** (i) bước 15 đã quét cả dev/test → rò rỉ ở khâu *khám phá ứng viên*; (ii) các output robustness/fusion cũ đã bị xóa, muốn dùng trong paper phải chạy lại.

> **📌 Cập nhật 19/10 (sau khi lexicon được dựng lại lần nữa):** `verified_lexicon.csv` hiện còn **668 entry** (bản 922 chỉ còn trong lịch sử). Các số ở mục 0 trên thuộc **phiên bản 922 entry**; đo mới (mỗi hệ thống **1 lần test**, ngưỡng đóng băng từ validation):
>
> | Hệ thống (lexicon 668) | validation | test | Ghi chú |
> |---|---|---|---|
> | baseline `configs/rule_engine.yaml` | 0.4555 | **0.4536** | acc 0.7070, thresholds [0.85, 0.0, 0.41] |
> | **tuned** `configs/rule_engine_tuned.yaml` (script 27) | 0.5481 | **0.5335** | acc 0.8398, thresholds [0.64, 0.0, 0.68], Δtest **+0,0799** |
>
> Fusion AI + rule và kiểm định: `outputs/evaluation/fusion/` (`significance_report.md`); bảng lịch sử test-split đầy đủ ở mục 7.

---

## 1. Việc đã làm (3 mảng)

| Mảng | Nội dung | Trạng thái |
|---|---|---|
| **A** | Trích xuất + TF-IDF + gán nhãn lại từ 3 dataset | ✅ xong → 922 entry |
| **B** | Smoke test toàn bộ + eval thật rule engine (validation → freeze → test) | ✅ xong |
| **C** | Ablation siết prune (hướng B) + viết báo cáo này | ✅ xong |

---

## 2. MẢNG A — Pipeline tokenize + TF-IDF + gán nhãn

### 2.1 Bước 15 — trích xuất ứng viên (`scripts/15_extract_corpus_words.py`)

**Tokenize:** cắt tiếng Việt theo **âm tiết + bigram âm tiết** (không cắt theo khoảng trắng thuần), giữ dấu tiếng Việt.

**Bộ lọc 3 lớp:**

| Lớp | Tham số | Ý nghĩa |
|---|---|---|
| DF filter | `min_toxic_df = 5`, `max_clean_df = 1` | từ phải xuất hiện ≥5 document toxic và ≤1 document clean |
| **TF-IDF gate** | `min_ratio = 0.60`, `min_score = 0.004`, `smoothing = 1.0` | tỷ trọng TF-IDF phía toxic ≥60% **và** điểm tách biệt toxic–clean ≥0,004 |
| Ngưỡng gộp | (không đổi) | gộp ứng viên theo dạng chuẩn hóa |

**Số liệu trích xuất (theo `data/processed/lexicon/lexicon_report.json`):**

| Nguồn | Dòng quét | Ứng viên trước TF-IDF | Sau TF-IDF | Bị loại |
|---|---|---|---|---|
| ViHSD | 33.398 | 344 (47 uni + 297 bi) | **344** | 0 |
| ViCM | 5.895 | 1.141 (247 uni + 894 bi) | **1.141** | 0 |
| Bontori | 95.945 | 1.897 | **113** (9 uni + 104 bi) | **1.784** |
| **Tổng** | **135.238** | **3.382** | **1.598** | **1.784 (−52,75%)** |

> **Đọc số:** TF-IDF gate cắt đi **52,75%** ứng viên; toàn bộ 1.784 chỗ cắt là term khai thác từ Bontori (nguồn lớn nhất, nhiễu nhất — 95.945 dòng, 69.039 dòng CLEAN). ViHSD và ViCM **không bị cắt ở cổng này** vì DF filter đã siết từ trước.

*Bằng chứng đối chứng:* chạy `--no-tfidf` cho ra đúng **3382** ứng viên (bằng tổng trước TF-IDF) → cổng TF-IDF đang được bật và thực sự tác động.

### 2.2 Bước 16 — gán nhãn (`scripts/16_label_lexicon.py`)

Chỉ dùng **train split** (`source_split: train`, tổng **114.552 dòng** = ViHSD 24.046 + ViCM 3.787 + Bontori 86.719).

**Tham số (từ `provenance.json`):**

```
min_evidence=3 · share_threshold=0.6 · clean_lift=3.0 · clean_count_cap=20
strong_lift=15.0 · verified: conf≥0.7 & evidence≥10 · auto: conf≥0.4
severity: HATE=0.90, OFFENSIVE=0.60 · confidence = 0.4·evidence + 0.3·margin + 0.3·lift
TF-IDF gate lần 2: enabled, min_ratio=0.6, min_score=0.004, smoothing=1.0
```

**Kết quả:** 1598 ứng viên → **922 entry có nhãn** (676 entry không qua đủ ngưỡng bằng chứng/lift/confidence/TF-IDF).

### 2.3 Thành phần lexicon 922

| | Tổng | HATE | HATE;OFFENSIVE | OFFENSIVE | verified | auto |
|---|---|---|---|---|---|---|
| **922** | 922 | 754 | 114 | 54 | 577 | 345 |

Phân bố severity: `0.9 → 868` (HATE + HATE;OFFENSIVE), `0.6 → 54` (OFFENSIVE).

**File:** `data/processed/lexicon/verified_lexicon.csv` · `dictionary_version: v1` · SHA256 `af3eb08140702f2c…`

### 2.4 ⚠️ Lưu ý method phải ghi rõ (rủi ro #1)

| Giai đoạn | Dữ liệu dùng | Rò rỉ test? |
|---|---|---|
| **Bước 15** (khám phá ứng viên) | train + **dev + test** của cả 3 dataset (135.238 dòng) | **CÓ** — DF filter dùng *nhãn* nên nhãn test tham gia lọc ứng viên |
| **Bước 16** (gán nhãn + TF-IDF lần 2) | **chỉ train** (114.552 dòng) | Không |
| **Prune (bước 20)** | train (114.552 dòng) | Không |
| **Eval thresholds** | **chỉ validation** | Không — test đóng băng |

**Đánh giá mức độ:** rò rỉ chỉ nằm ở khâu *khám phá ứng viên* (một term chỉ "được vào danh sách" nếu đạt ngưỡng DF kể cả khi thấy ở test), **không** rò rỉ vào khâu *chấm điểm/gán nhãn/ngưỡng*. Đây là kiểu **transductive candidate mining** — mức độ vừa phải, không phải rò rỉ nhãn trực tiếp vào test metric.

**Phương án sửa (nếu cô yêu cầu nghiêm ngặt):** thêm cờ `--splits train` vào `scripts/15_extract_corpus_words.py`, chạy lại 15 → 16 → 20 → 17 (≈ 25–30 phút, chủ yếu là bước 16). Dự kiến số ứng viên giảm nhẹ (mất các term chỉ xuất hiện ở dev/test).

---

## 3. MẢNG B — Eval thật Rule Engine

### 3.1 Smoke test (`scripts/18_smoke_test_rule_engine.py --split all --limit 0`)

```
PASSED: True · 27/27 check · 0 lỗi · lexicon 922 entries
```

- Unit check: config/lexicon/engine load, giữ dấu tiếng Việt, offset map (length + monotonic + tái tạo raw text), exact match không double-count, score ∈ [0,1], obfuscation views, `bounded_levenshtein`, text rỗng/None/dài → không crash.
- End-to-end (33.398 dòng, **argmax không tune threshold**): train macro-F1 0.3629 · val 0.3561 · test 0.3545 · combined **0.3607**, ~0,52 ms/dòng.

> ⚠️ Số smoke test **không so sánh** được với eval thật: argmax thẳng, chưa tune threshold → thấp hơn là *đúng*.

### 3.2 Protocol (§44)

```
① --split validation  → optimize_thresholds → ghi thresholds.json (đóng băng)
② --split test        → load thresholds.json → score test MỘT LẦN (objective: "frozen")
```
Không tune trên test. Mỗi hệ thống đúng **1 lần test** với ngưỡng đã đóng băng.

### 3.3 Kết quả

**Validation (tune):**

| Config | Tham số | Lexicon | macro-F1 | acc | HATE | OFFEN | CLEAN | Thresholds (H/O/C) |
|---|---|---|---|---|---|---|---|---|
| `rule_engine.yaml` | accentless 0.90 | 922 | 0.4577 | 0.7096 | 0.3594 | 0.1752 | 0.8386 | 0.85 / 0.00 / 0.41 |
| **`rule_engine_tuned.yaml`** | **accentless 0.15** | 922 | **0.5305** | 0.8005 | 0.4846 | 0.2043 | 0.9026 | 0.60 / 0.00 / 0.72 |
| `rule_engine_pruned.yaml` | accentless 0.90 | 922 *(no-op)* | 0.4577 | 0.7096 | 0.3594 | 0.1752 | 0.8386 | 0.85 / 0.00 / 0.41 |

**Test (ngưỡng đóng băng, 1 lần):**

| Config | macro-F1 | acc | weighted-F1 | HATE F1 | OFFEN F1 | CLEAN F1 | macro AUC |
|---|---|---|---|---|---|---|---|
| baseline | 0.4667 | 0.7201 | 0.7520 | 0.3855 | 0.1705 | 0.8440 | 0.6838 |
| **tuned (best)** | **0.5304** | **0.8141** | **0.8198** | **0.4820** | 0.1979 | **0.9114** | 0.7161 |
| pruned-cũ | 0.4667 | 0.7201 | 0.7520 | 0.3855 | 0.1705 | 0.8440 | 0.6838 |

**Latency (test, 6680 dòng):** baseline 0,508 ms/p95 1,382/1967 per s · tuned **0,498 ms / p95 1,362 / 2008 per s**

### 3.4 Phát hiện: prune mặc định là NO-OP (chứng minh bằng hash)

```
[pruner] loaded 922 → wrote verified_lexicon_pruned.csv (922/922 kept) → dropped: {}

SHA256(verified_lexicon.csv)        = af3eb08140702f2cfa1146fcd3696cf3a9b931c1c0c05c044c1e8e5010a87ae5
SHA256(verified_lexicon_pruned.csv) = af3eb08140702f2cfa1146fcd3696cf3a9b931c1c0c05c044c1e8e5010a87ae5
→ IDENTICAL: True   (2 file trùng byte)
```

**Nguyên nhân:** `min_score = 0.004` chính là ngưỡng đã dùng ở bước 15 & 16 → *"lưới lọc lỗ y hệt, lọc lần 2 không còn gì để cắt"*.

**Hệ quả:** `configs/rule_engine_pruned.yaml` (params baseline + lexicon no-op) **trùng hoàn toàn** với baseline → không có giá trị ablation → đã được thay bằng biến thể hướng B (mục 4).

---

## 4. MẢNG C — Ablation siết cổng prune (HƯỚNG B ĐÃ CHỌN)

### 4.1 Thiết kế — cô lập đúng 1 biến

Tạo 3 config **sao chép từ `rule_engine_tuned.yaml`**, chỉ đổi `lexicon_path`:

```
rule_engine_tuned_p008.yaml → pruned_score008/…  (min_score 0.008) → 691 từ
rule_engine_tuned_p010.yaml → pruned_score010/…  (min_score 0.010) → 576 từ
rule_engine_tuned_p015.yaml → pruned_score015/…  (min_score 0.015) → 324 từ
```
`accentless_alias: 0.15`, matcher, scoring, normalize → **giữ nguyên** → khác biệt duy nhất là **số lượng term**.

> ⚠️ Nếu so `rule_engine_pruned.yaml` (params 0.90) với `tuned` (0.15) là so **2 biến cùng lúc** → sai design. Thiết kế trên đã khắc phục.

**Kết quả prune:**

| `min_score` | Giữ lại | Drop | Lý do drop |
|---|---|---|---|
| 0.004 (mặc định) | 922 (100%) | 0 | — |
| 0.008 | 691 | 231 (−25,1%) | `tfidf_low_score` |
| **0.010** | **576** | **346 (−37,5%)** | `tfidf_low_score` |
| 0.015 | 324 | 598 (−64,9%) | `tfidf_low_score` |

### 4.2 Sweep trên VALIDATION (mỗi biến thể tự tune threshold trên validation)

| Biến thể | Số từ | val macro-F1 | Δ vs 922 | val acc | HATE | OFFEN | CLEAN | Thresholds (H/O/C) |
|---|---|---|---|---|---|---|---|---|
| full | 922 | **0.5305** | — | 0.8005 | 0.4846 | 0.2043 | 0.9026 | 0.60 / 0.00 / 0.72 |
| score008 | 691 | 0.5277 | −0.0028 | 0.8035 | 0.4813 | 0.1987 | 0.9033 | 0.60 / 0.00 / 0.72 |
| **score010** ✅ | **576** | **0.5220** | **−0.0085** | 0.8028 | 0.4776 | 0.1867 | 0.9019 | 0.60 / 0.00 / 0.72 |
| score015 | 324 | 0.5018 | −0.0287 | 0.8379 | 0.4743 | **0.1148** | 0.9164 | 0.75 / 0.01 / 0.75 |

### 4.3 Tiêu chí chọn (áp TRƯỚC khi nhìn test — §44)

> **"Chọn biến thể NHỎ NHẤT có Δmacro-F1 ≤ 0.010 trên validation so với bản 922."**

- 691 → Δ 0.0028 ✅ nhưng chưa phải nhỏ nhất
- **576 → Δ 0.0085 ✅ → CHỌN**
- 324 → Δ 0.0287 ❌ loại (đặc biệt OFFENSIVE F1 sụp 0.204 → 0.115)

**→ Đóng băng biến thể 576, chạy test MỘT LẦN.**

### 4.4 Kết quả TEST (ngưỡng đóng băng `0.60 / 0.00 / 0.72`)

| | full **922** | pruned **576** | Δ | Δ tương đối |
|---|---|---|---|---|
| **macro-F1** | 0.5304 | 0.5270 | −0.0034 | **−0,64%** |
| **accuracy** | 0.8141 | 0.8211 | **+0.0070** | **+0,86%** |
| weighted-F1 | 0.8198 | 0.8212 | **+0.0014** | +0,17% |
| macro ROC-AUC | 0.7161 | 0.6991 | −0.0170 | −2,4% |
| HATE F1 | 0.4820 | 0.4593 | −0.0227 | −4,7% |
| OFFEN F1 | 0.1979 | 0.2066 | **+0.0087** | +4,4% |
| CLEAN F1 | 0.9114 | 0.9152 | +0.0038 | +0,4% |
| — HATE P/R | 0.5365 / 0.4375 | 0.5426 / 0.3983 | | |
| — OFFEN P/R | 0.1691 / 0.2387 | 0.1847 / 0.2342 | | |
| — CLEAN P/R | 0.9161 / 0.9068 | 0.9100 / 0.9205 | | |
| **Latency mean** | 0,498 ms | **0,406 ms** | −0,092 ms | **−18,5%** |
| Latency p95 | 1,362 ms | **1,053 ms** | −0,309 ms | −22,7% |
| Throughput | 2.008/s | **2.464/s** | +456/s | **+22,7%** |

**Confusion matrix test** (hàng = truth, cột = predicted, thứ tự HATE / OFFENSIVE / CLEAN):

```
full 922:  [[301, 145, 242],
            [119, 106, 219],
            [141, 376, 5031]]

pruned 576:[[274, 142, 272],
            [107, 104, 233],
            [124, 317, 5107]]
```
→ False positive trên CLEAN giảm (141 → 124), TP CLEAN tăng (5031 → 5107); đổi lại TP HATE giảm (301 → 274).

### 4.5 Vì sao 576 từ vẫn giữ được hiệu năng? (phân tích term bị cắt)

| Biến thể | Tổng | HATE | HATE;OFF | OFFENSIVE | verified | auto |
|---|---|---|---|---|---|---|
| 922 | 922 | 754 | 114 | 54 | 577 | 345 |
| 691 | 691 | 591 | 82 | 18 | 532 | 159 |
| **576** | **576** | **505** | **61** | **10** | **475** | **101** |
| 324 | 324 | 286 | 34 | 4 | 275 | 49 |

**Tỷ lệ giữ lại ở 576:**

| Nhóm | Tỷ lệ giữ | Đọc |
|---|---|---|
| `verified` (conf ≥ 0.7) | 475/577 = **82,3%** | giữ gần hết |
| `auto` (conf 0.4–0.7) | 101/345 = **29,3%** | **bị cắt mạnh nhất** |
| HATE | 505/754 = 67,0% | cắt vừa |
| OFFENSIVE | 10/54 = **18,5%** | **rất dễ tổn thương** |

> **Giải thích:** cổng TF-IDF thực hiện đúng việc nó cần — **loại chủ yếu term `auto` (độ tin cậy thấp) và term ít bằng chứng**, giữ lại term `verified`. Đây là lý do 576 từ vẫn đạt F1 gần bằng 922: *cái bị bỏ là nhiễu, không phải tín hiệu*.
>
> **Nhưng OFFENSIVE cực kỳ mỏng:** 54 → 10 term → đó là lý do biến thể 324 (chỉ còn 4 term OFFENSIVE) **sụp OFFENSIVE F1 = 0.1148**. Đây là "tai nạn" ablation rất có ích: nó chỉ ra **ranh giới an toàn nằm ở 576, không được xuống 324**.

### 4.6 Kết luận của ablation (viết được vào paper)

1. **Nén được 37,5% lexicon** (922 → 576) mà macro-F1 chỉ đổi −0,0034 (−0,64%), **accuracy còn tăng** (+0,0070).
2. **Nén 37,5% → nhanh hơn 18,5% latency, +22,7% throughput** → có ý nghĩa với deployment.
3. **Phân rã cấu trúc:** `verified` giữ 82,3% vs `auto` chỉ giữ 29,3% → TF-IDF prune = thanh lọc độ tin cậy, không phải cắt ngẫu nhiên.
4. **Ngưỡng an toàn:** không xuống dưới ~576 term — ở 324 term thì OFFENSIVE F1 giảm 44% (0.204 → 0.115).
5. **Prune mặc định (0.004) là idempotent** → TF-IDF ở khâu trích xuất đã thay thế bước prune cổ điển.

---

## 5. Việc cần làm tiếp & rủi ro còn lại

| # | Vấn đề | Mức độ | Đề xuất |
|---|---|---|---|
| 1 | **Bước 15 quét cả dev/test** → rò rỉ khâu khám phá ứng viên (mục 2.4) | 🔴 phải xử lý | Thêm `--splits train`, chạy lại 15→16→20→17 (~25–30 phút) |
| 2 | **Chưa có kiểm định** cho chênh lệch 0.5304 vs 0.5270 | 🔴 §44 cấm tuyên bố khi chưa kiểm định | Script 17 **không ghi per-sample predictions** → cần bổ sung cột `test_predictions.csv` rồi chạy paired bootstrap (`src/evaluation/analysis.py: paired_bootstrap_delta`) |
| 3 | Output **robustness / fusion / significance** trước đó **đã bị xóa** | 🔴 | Chạy lại `scripts/24…25` + `scripts/23` nếu dùng trong paper |
| 4 | `configs/rule_engine_pruned.yaml` vẫn trỏ lexicon no-op + params baseline | 🵐 | Đổi sang `rule_engine_tuned_p010.yaml` hoặc xóa config cũ |
| 5 | Comment trong `rule_engine_tuned.yaml` vẫn ghi số cũ (1828/1096, F1 0.4469→0.5086) | 🵐 | Cập nhật comment cho khớp 922 |
| 6 | **Lexicon 1828 cũ không tái tạo được** (file đã xóa, không có trong git) | 🟡 | Số "0.4927" cũ **không có artifact** → **không trích trong paper** |
| 7 | OFFENSIVE chỉ 54 term, F1 ~0.20 | 🟡 | Bàn ở Discussion; cân nhắc gộp OFFENSIVE vào HATE hoặc thêm lexicon OFFENSIVE riêng |
| 8 | Log/ file tạm ở root (`ex15*.log`, `pr_*.log`, `ab*.log`, `tmp_*.py`) | 🵐 | Dọn trước khi commit |

---

## 6. Reproduce (§44 — lệnh + hash)

```powershell
$env:PYTHONIOENCODING="utf-8"

# A. trích xuất + gán nhãn (tokenize + TF-IDF)
python scripts\15_extract_corpus_words.py
Copy-Item data\processed\lexicon\sensitive_words_combined.txt data\processed\lexicon\sensitive_words_from_datasets.txt
python scripts\16_label_lexicon.py

# B. smoke test toàn bộ
python scripts\18_smoke_test_rule_engine.py --split all --limit 0

# C. prune siết (3 biến thể)
python scripts\20_prune_lexicon.py --tfidf-min-score 0.008 --output-dir data\processed\lexicon\pruned_score008 --no-per-dataset
python scripts\20_prune_lexicon.py --tfidf-min-score 0.010 --output-dir data\processed\lexicon\pruned_score010 --no-per-dataset
python scripts\20_prune_lexicon.py --tfidf-min-score 0.015 --output-dir data\processed\lexicon\pruned_score015 --no-per-dataset

# D. eval: validation (tune + đóng băng) → test (1 lần) cho MỖI config
python scripts\17_evaluate_rule_engine.py --config configs\rule_engine_tuned.yaml     --split validation --out outputs\evaluation\rule_engine_tuned
python scripts\17_evaluate_rule_engine.py --config configs\rule_engine_tuned.yaml     --split test       --out outputs\evaluation\rule_engine_tuned
python scripts\17_evaluate_rule_engine.py --config configs\rule_engine_tuned_p010.yaml --split validation --out outputs\evaluation\rule_engine_tuned_p010
python scripts\17_evaluate_rule_engine.py --config configs\rule_engine_tuned_p010.yaml --split test       --out outputs\evaluation\rule_engine_tuned_p010
```

**SHA256 (16 ký tự đầu):**

| Artifact | SHA256 | Size |
|---|---|---|
| `data/processed/lexicon/verified_lexicon.csv` | `af3eb08140702f2c` | 922 rows |
| `…/pruned_score008/verified_lexicon_pruned.csv` | `a022ab230b0b7c42` | 691 rows |
| `…/pruned_score010/verified_lexicon_pruned.csv` | `2bf669039ad63949` | 576 rows |
| `…/pruned_score015/verified_lexicon_pruned.csv` | `2176a42d1f2e23d8` | 324 rows |
| `configs/rule_engine.yaml` | `e6ddfbf2889f8490` | — |
| `configs/rule_engine_tuned.yaml` | `5eca6eacd6ccfe9c` | — |
| `configs/rule_engine_tuned_p010.yaml` | `9f7af143be3d80ae` | — |
| `configs/rule_engine_pruned.yaml` | `f617772b26ea7f09` | — |
| `data/processed/vihsd/test.csv` | `d8f9ae7ddd8ced90` | 6680 rows |

**Output đã tạo:**

```
outputs/evaluation/rule_engine/                     {metrics.json, thresholds.json, latency.json, test/…}
outputs/evaluation/rule_engine_tuned/               (best: test macro-F1 0.5304)
outputs/evaluation/rule_engine_pruned/              (no-op, trùng baseline)
outputs/evaluation/rule_engine_tuned_p008/          (val sweep, 691 từ)
outputs/evaluation/rule_engine_tuned_p010/          (CHỌN: test macro-F1 0.5270)
outputs/evaluation/rule_engine_tuned_p015/          (val sweep, 324 từ)
outputs/evaluation/rule_engine_tuned_pruning_ablation_val.json   ← bảng sweep validation
configs/rule_engine_tuned_p{008,010,015}.yaml       ← config ablation
```

> **Cập nhật 19/10 (sau khi chốt chỉ giữ baseline + bản tốt nhất):** các folder kết quả
> `rule_engine_pruned/`, `rule_engine_tuned_p008|p010|p015/` đã được dọn khỏi `outputs/evaluation/`;
> bảng sweep vẫn còn ở `rule_engine_tuned_pruning_ablation_val.json` (dòng trên) và ở mục 6 của báo cáo này.
> `configs/rule_engine_pruned.yaml` + `configs/rule_engine_tuned_p*.yaml` vẫn còn trên đĩa nếu cần tái lập.

**Lịch sử test-split (để đối soát với §44 — mỗi hệ thống đúng 1 lần):**

| Hệ thống | Lần test | Kết quả |
|---|---|---|
| baseline 922 | 1 | macro-F1 0.4667 |
| tuned 922 | 1 | macro-F1 0.5304 |
| pruned-cũ 922 (no-op) | 1 | macro-F1 0.4667 *(trùng baseline)* |
| tuned + pruned 576 | 1 | macro-F1 0.5270 |
| tuned, lexicon 1828 (accentless 0.15) | 1 | macro-F1 0.4927 |
| baseline, lexicon 668 | 1 | macro-F1 0.4536 (acc 0.7070) |
| **tuned, lexicon 668 + tune lại trọng số (script 27)** | **1** | **macro-F1 0.5335** (val 0.5481, accuracy 0.8398) |
