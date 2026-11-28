# README — Đánh giá bài báo Hybrid (mDeBERTa + Rule Engine)

> Tài liệu này tóm tắt điểm mạnh, điểm yếu và kế hoạch cải thiện cho bài báo hướng
> **Hybrid = Model AI + Rule Engine**, chuẩn bị nộp ICCIES 2027 (Springer CCIS,
> Scopus, 12–15 trang, LNCS style). Đọc từ trên xuống: mục 1–2 cho bức tranh tổng,
> mục 3–4 là việc cần làm, mục 5–6 là kế hoạch và file tham chiếu.

---

## 1. Tóm tắt đánh giá

| | |
|---|---|
| **Điểm hiện tại** | **6,5 / 10** — đủ nộp nhưng chưa chắc ăn |
| **Điểm tiềm năng sau khi sửa mục BẮT BUỘC** | **7,5–8 / 10** |
| **Khả năng được chấp nhận ở ICCIES** (acp rate ~45–50%) | Viết đúng framing: ~55–65%. Viết sai framing (đòi "vượt AI"): ~25–35% |
| **Vấn đề cốt lõi** | Hybrid **không vượt** model AI tốt nhất (0.6647 vs 0.6638 — hòa trong nhiễu). Giá trị thật nằm ở **giải thích được + chống lách luật**, không phải ở độ chính xác |

---

## 2. Điểm tốt — giữ nguyên và phát huy

1. **Phương pháp luận chặt chẽ, hiếm thấy ở cấp CCIS**
   - Tune trọng số rule engine **chỉ trên validation** (155 trial ghi đầy đủ trong `tuning_trials.csv`), đóng băng rồi mới chấm test đúng 1 lần.
   - Kiểm định thống kê đầy đủ: paired bootstrap 10.000 lần + McNemar chính xác.
   - Từ điển có SHA-256 provenance, chỉ trích xuất từ **train split** (chống rò rỉ dữ liệu).

2. **Rule engine cải thiện +0,0799 test macro-F1** (0.4536 → 0.5335) sau khi tune trọng số — bản thân đây là một kết quả đáng kể, độc lập với fusion.

3. **Khả năng chống lách luật là điểm khác biệt thật sự**
   - Phát hiện viết né lọc (leet / chèn khoảng trắng / lặp ký tự) mà transformer hay bỏ sót.
   - Số liệu robustness hiện có: leet −0.02, spaced +0.115, repeat +0.048 (so với clean).

4. **Giải thích được (explainability)** — AI thuần không có: vị trí khớp, dạng gốc/normalized, quy tắc kích hoạt, điểm số từng match, quyết định ALLOW / REVIEW / BLOCK.

5. **Trung thực về kết quả âm tính** — đã báo cáo đầy đủ các hướng thất bại (3/4 hướng cải thiện lexicon thất bại, fusion + NB không ý nghĩa). Điều này tạo tín nhiệm với reviewer.

6. **Phù hợp chủ đề ICCIES** — main track "Intelligent Information Systems / Tools and Applications" khớp trực tiếp với hệ thống kiểm duyệt có giải thích.

7. **Tính tái lập cao** — config đóng băng, `predictions.csv` từng mẫu, `rule_scores.csv`, threshold đóng băng, đủ để ai cũng kiểm tra lại được.

---

## 3. Điểm chưa tốt — cần cải thiện

### MỨC 1 — BẮT BUỘC (quyết định đậu / rớt)

**1.1 Đổi tuyên bố chính của bài.**
Không được nói "hybrid vượt model AI" — số liệu không cho phép. Tuyên bố thay thế gồm 3 vế:
(a) **ngang bằng** model AI tốt nhất (trong khoảng nhiễu thống kê),
(b) **giải thích được** (bằng chứng, quy tắc, quyết định),
(c) **chống lách luật** (robustness trước viết né lọc).
Ghi rõ "parity" ngay trong abstract để reviewer không đi tìm con số SOTA.

**1.2 Tự nêu 2 kết quả âm tính trước khi reviewer tìm ra.**
- Fusion làm model `sensitiveai-vi-stage2` **tệ hơn có ý nghĩa** (Δ = −0.0085, McNemar p = 0.049).
- Rule engine sau khi tune **tụt mạnh ở `no_accent`** (−0.1175 so với clean, vì `accentless_alias` bị đưa về 0).
Giấu hai điểm này = rủi ro bị từ chối. Trình bày thẳng + phân tích nguyên nhân = điểm cộng về tín nhiệm.

**1.3 Bổ sung phân tích "khi nào rule engine giúp" — quan trọng nhất trong toàn bộ danh sách.**
Dữ liệu gốc đã có sẵn trong supplementary analysis:
- **504 comment bị cả 6 model AI bỏ sót** (185 HATE, 180 OFFENSIVE, 139 CLEAN).
Việc cần làm:
- Test xem rule engine tuned có bắt được phần nào trong 504 mẫu đó không. Có → câu chuyện mạnh ("RE phủ lỗ hổng chung của AI"). Không → phải ghi nhận như limitation.
- Phân tích disagreement giữa AI và RE trên toàn test: RE-sửa-AI bao nhiêu mẫu, AI-sửa-RE bao nhiêu, loại lỗi nào (dùng `false_positive_analysis.csv` / `false_negative_analysis.csv` đã có).

**1.4 Tính calibration cho đầu ra fusion.**
Hiện ECE/Brier/MCC chỉ có cho 6 model AI (`outputs/evaluation/analysis/supplementary_report.md`), **chưa có cho hybrid**. Reviewer sẽ hỏi "điểm hybrid có calibrated không?". Cần chạy thêm 1 script tính ECE cho fusion vs AI-only.

**1.5 Định vị rõ so với literature tiếng Việt.**
Trích dẫn PhoBERT-CNN, ViHateT5, ViSoBERT, ViHOS (đã có trong `papers/references.bib`) và chỉ ra khoảng trống: **không bài nào cung cấp bằng chứng giải thích + chống lách luật** — đó là chỗ đứng của bài này.

### MỨC 2 — NÊN LÀM (nâng điểm lên ~7,5)

**2.1 Trả lời trước câu hỏi "RE yếu (0.53) sao còn fuse?"**
Vai trò của RE không phải độ chính xác, mà là: bằng chứng, phủ lỗ hổng obfuscation, và định tuyến REVIEW cho các ca mơ hồ.

**2.2 Giải thích vì sao chỉ model curriculum có fusion ý nghĩa.**
+0.0045 với p = 0.033 (bootstrap) và McNemar p = 0.0005 — cần cơ chế: có thể lỗi của curriculum thiên về lexicon-correctable. Phân tích ở mục 1.3 sẽ cung cấp cơ chế này.

**2.3 Báo cáo latency tổng hợp end-to-end.**
Đã có `latency.json` của RE và benchmark của AI — gộp thành một bảng chi phí hybrid. Quan trọng với hội nghị ứng dụng.

**2.4 Decision layer (ALLOW/REVIEW/BLOCK).**
Nếu có số liệu mô phỏng định tuyến REVIEW thì đưa vào; nếu chưa có thì **chuyển sang Future Work** để không bị đánh giá là tuyên bố quá mức.

**2.5 Xử lý vấn đề no_accent.**
Hai lựa chọn: (a) trình bày như trade-off có chủ đích kèm bảng robustness đầy đủ; hoặc (b) chạy thử lại nhanh với `accentless_alias` ở mức trung gian (0.2–0.4) xem có giữ được no_accent mà không mất leet/spaced/repeat không.

### MỨC 3 — NÊN CÓ (nâng lên ~8)

**3.1 Ensemble.** Cân nhắc thêm subsection ngắn dùng kết quả trong `outputs/evaluation/ensemble/` (vote 3 model + RE) — chỉ nếu nó bổ sung một điểm đáng kể, đừng làm loãng câu chuyện chính.

**3.2 Threats to validity.** Thêm "chỉ 1 dataset (ViHSD)" và (nếu có đo) độ đồng thuận gán nhãn từ điển (plan §43 yêu cầu mục này).

**3.3 Đánh giá ngoài ViHSD.** Nếu có thời gian, chạy hybrid trên một tập tiếng Việt nhỏ bên ngoài (VD: mẫu từ ViHOS hoặc thu thập thủ công) để có bằng chứng khái quát hóa.

---

## 4. Số liệu chính cần nhớ khi viết

| Chỉ số | Giá trị |
|---|---|
| AI tốt nhất (M3, AI-only) | macro-F1 **0.6638** |
| Hybrid tốt nhất (curriculum + per-channel thresholds) | macro-F1 **0.6647** |
| Fusion có ý nghĩa duy nhất | curriculum: +0.0045, bootstrap p=0.033, McNemar p=0.0005 |
| Fusion làm tệ hơn (cần nêu) | stage2: −0.0085, McNemar p=0.049 |
| Rule engine baseline → tuned (test) | 0.4536 → **0.5335** (+0.0799) |
| Rule engine tuned (validation) | 0.5481 |
| Robustness (tuned, so với clean) | leet −0.02 · spaced +0.115 · repeat +0.048 · **no_accent −0.1175** |
| Mẫu bị cả 6 AI bỏ sót | **504** (185 HATE, 180 OFFENSIVE, 139 CLEAN) |
| Fusion + NB | +0.0030, không ý nghĩa |

---

## 5. Kế hoạch hành động (theo thứ tự)

1. **Phân tích blind-spot** (mục 1.3): script đọc `predictions.csv` + `rule_scores.csv` đã có, đếm RE bắt được bao nhiêu trong 504 mẫu, và bảng disagreement AI↔RE.
2. **Calibration fusion** (mục 1.4): tính ECE/Brier cho hybrid vs AI-only.
3. **Viết lại abstract** theo 3 tuyên bố parity + explainability + robustness.
4. **Viết section "When does the rule engine help?"** từ kết quả bước 1 — đây là section quyết định.
5. **Viết section Results** dẫn dắt bằng 2 kết quả âm tính (mục 1.2) trước khi nêu điểm mạnh.
6. **Related work** theo mục 1.5.
7. **Threats to validity** theo mục 3.2.
8. Kiểm tra độ dài 12–15 trang theo template `llncs` (Springer CCIS).
9. Kiểm tra chính sách **AI-generated text** của ICCIES trước khi nộp camera-ready.

---

## 6. File tham chiếu chính

| Nội dung | Đường dẫn |
|---|---|
| Kế hoạch rule engine gốc | `Rule_Engine_Plan.md` (mục 35–44) |
| Kết quả fusion tổng hợp | `outputs/evaluation/fusion/comparison.md` |
| Kiểm định fusion chi tiết | `outputs/evaluation/fusion/significance_report.md`, `fusion_significance.json` |
| Rule engine tuned | `outputs/evaluation/rule_engine_tuned/` (metrics, robustness, tuning_trials.csv) |
| Robustness | `outputs/evaluation/rule_engine_tuned/test/robustness.json` |
| Calibration 6 model AI | `outputs/evaluation/analysis/supplementary_report.md` |
| Error complementarity (504 mẫu) | `outputs/evaluation/analysis/supplementary_report.md` (mục "Error complementarity") |
| Các hướng cải thiện lexicon đã thử | `outputs/lexicon_experiments/directions_summary.md` |
| Ensemble | `outputs/evaluation/ensemble/` |
| References sẵn có | `papers/references.bib` |
| Template Springer | `papers/main.tex` (llncs, 12–15 trang) |
