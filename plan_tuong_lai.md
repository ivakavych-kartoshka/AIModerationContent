# KẾ HOẠCH PHÁT TRIỂN DỰ ÁN SENSITIVEAI — 2026–2028

> Lộ trình tiếp theo của dự án AIModerationContent, được lập ngày 2026-10-21 sau khi
> rà soát toàn bộ kết quả thực nghiệm. Mục tiêu: nộp thành công bài hybrid cho
> **ICCIES 2027** (rank C/Scopus), đồng thời xây gói **benchmark chống lách luật
> tiếng Việt** để nộp **IEEE SMC 2027** (CORE B, tổ chức tại TP.HCM 6–10/10/2027,
> deadline 15/04/2027) và sau đó là tạp chí Q2–Q3. Tài liệu này thay thế mọi kế
> hoạch ngắn hạn trước đó về phần viết bài.

---

## 0. Tổng quan

| Giai đoạn | Thời gian | Mục tiêu | Sản phẩm |
|---|---|---|---|
| **GĐ1 — Nộp ICCIES 2027** | 10/2026 – 12/2026 | Chấp nhận bài hybrid (model + rule engine) | Bài 12–15 trang, Springer CCIS |
| **GĐ2 — Xây benchmark** | Song song → 03/2027 | Benchmark "chống lách luật tiếng Việt" + lexicon công khai | Repo GitHub/HF + báo cáo kỹ thuật |
| **GĐ3 — Nộp IEEE SMC 2027** | 03–07/2027 | Chấp nhận bài benchmark ở CORE B | Bài IEEE 6 trang, IEEE Xplore |
| **GĐ4 — Journal (tùy chọn)** | 2027–2028 | Mở rộng thành bài tạp chí Q2–Q3 | Journal paper |

Nguyên tắc xuyên suốt (giữ từ kế hoạch gốc): **chấm test đúng 1 lần, mọi lựa chọn
chỉ từ train/validation, đóng băng trước khi đánh giá, kiểm định thống kê đầy đủ,
báo cáo cả kết quả âm tính.**

---

## 1. GIAI ĐOẠN 1 — Nộp bài Hybrid cho ICCIES 2027

### 1.1. Việc khẩn cấp (làm ngay trong tuần này)

- [ ] **Xác nhận deadline ICCIES 2027** — kiểm tra https://iccies.tdtu.edu.vn/2027/ và
      CMT3 (https://cmt3.research.microsoft.com/ICCIES2027). Edition 2026 có deadline
      ~tháng 10–12, nên **deadline 2027 có thể rất gần** (có thể chỉ còn vài tuần).
- [ ] **Xác nhận bài chưa nộp ICEIR 2026** (deadline 15/10/2026 đã qua). Nếu đã nộp
      ICEIR → KHÔNG được nộp trùng ICCIES (vi phạm dual submission, Springer dùng
      iThenticate). Nếu chưa nộp → tiếp tục.
- [ ] Cập nhật phần header `papers/main.tex` từ "ICEIR 2026" sang "ICCIES 2027"
      (template `llncs` dùng chung, không cần đổi class).

### 1.2. Việc bắt buộc trước khi nộp (theo `README_danh_gia_hybrid.md`, mức 1)

1. [ ] **Phân tích blind-spot 504 mẫu** — viết script đọc `predictions.csv` của 6 model
      + `rule_scores.csv` của rule engine tuned, đếm:
      - RE bắt được bao nhiêu trong 504 mẫu mà cả 6 AI cùng sai (185 HATE, 180 OFFENSIVE)?
      - Bảng disagreement AI ↔ RE: RE-sửa-AI bao nhiêu mẫu, AI-sửa-RE bao nhiêu, loại lỗi nào.
      → Đây là section "When does the rule engine help?" — quan trọng nhất của bài.
2. [ ] **Calibration fusion** — tính ECE/Brier cho hybrid vs AI-only (hiện chỉ có cho
      6 model AI trong `outputs/evaluation/analysis/supplementary_report.md`).
3. [ ] **Viết lại abstract theo 3 tuyên bố**: (a) parity với AI tốt nhất, (b) giải thích
      được, (c) chống lách luật. Không được tuyên bố "vượt AI".
4. [ ] **Tự nêu 2 kết quả âm tính**: fusion làm `stage2` tệ hơn (McNemar p=0.049) và RE
      tụt ở `no_accent` (−0.1175) — kèm phân tích nguyên nhân.
5. [ ] **Bổ sung tài liệu tham khảo bắt buộc** vào `papers/references.bib`:
      - Ha Van Muon et al. 2026 — *EHF: Explainable Hybrid Framework* (DOI 10.54654/isj.v3i26.1162)
      - Evasion-Hate 2026 (DTA journal, augmentation chống evasion)
      - Gröndahl et al. — *All You Need is "Love"* (adversarial hate speech)
      - Viết đoạn phân biệt: ante-hoc (rule engine minh bạch) vs post-hoc (SHAP của EHF);
        ViHSD chuẩn vs dataset riêng của EHF; ý nghĩa của 92.5% F1 của EHF là khác
        label distribution, không so sánh trực tiếp với macro-F1 0.66 của mình.
6. [ ] **Chèn số liệu chính** theo bảng "Số liệu chính cần nhớ khi viết" trong
      `README_danh_gia_hybrid.md` mục 4.

### 1.3. Việc nên làm (nếu thời gian cho phép)

- [ ] Bảng latency end-to-end của hybrid (RE `latency.json` + AI benchmark).
- [ ] Xử lý `no_accent`: trình bày như trade-off, hoặc thử lại `accentless_alias`
      ở 0.2–0.4 xem có giữ được cả hai phía không.
- [ ] Decision layer: nếu có số liệu mô phỏng REVIEW thì đưa vào; không thì chuyển
      Future Work.
- [ ] Subsection ngắn về ensemble (chạy lại scripts 31–33, folder
      `outputs/evaluation/ensemble/` đã bị xóa) — chỉ để củng cố thông điệp
      "bottleneck nằm ở data, không phải aggregation".

### 1.4. Checklist trước khi bấm nút nộp

- [ ] 12–15 trang theo `llncs` (CCIS), tiếng Anh
- [ ] Điền ORCID, tên trường đúng (FPT University), acknowledgements, data availability
- [ ] Rà soát chính sách **AI-generated text** của ICCIES — văn bản phải là ngôn ngữ
      của tác giả, không để lộ văn phong máy
- [ ] Nộp qua CMT3 đúng deadline

**Tiêu chí hoàn thành GĐ1:** bài đã nộp + email xác nhận từ CMT.

---

## 2. GIAI ĐOẠN 2 — Xây gói benchmark "Chống lách luật tiếng Việt" (song song)

Mục tiêu: biến thực nghiệm hiện có thành **tài nguyên công khai** đủ chuẩn rank B.

### 2.1. Đóng gói benchmark (Q4 2026)

- [ ] **Định nghĩa benchmark v1**: các phép biến đổi đã có (leet / spaced / repeat /
      no_accent) áp dụng trên **ViHSD test**, giữ nhãn gốc, kèm metadata từng mẫu
      (transform type, nguồn mẫu gốc, offset).
- [ ] **Kiểm tra label drift**: xác minh phép biến đổi không làm đổi nghĩa/nhãn
      (lấy mẫu kiểm thủ công, ghi tỷ lệ đồng thuận) — reviewer rank B sẽ hỏi điều này.
- [ ] **Phân tích độ khó benchmark**: tỷ lệ phát hiện theo từng transform, theo lớp,
      theo độ dài comment — để tạo "difficulty profile".
- [ ] **Quyết định phân phối**: ViHSD là CC BY-NC 4.0 (kiểm tra lại) → nếu không phân
      phối lại được dữ liệu biến đổi, phát hành **code sinh benchmark** kèm hướng dẫn.

### 2.2. Baselines cho benchmark (Q1 2027)

- [ ] **Multi-seed**: chạy 1–2 hệ thống chính của mình (vd: M3 + M6) với **≥3 seeds**,
      báo cáo mean ± std (hiện tại mới có seed 42 — đây là lỗ hổng lớn nhất cho rank B).
      Ước tính ~10h/run trên RTX 4060 8GB → tổng ~30–60h, trải đều vài tuần.
- [ ] **Public models (inference/fine-tune)**: PhoBERT, ViSoBERT, ViHateT5 đánh giá
      trên benchmark (inference rẻ; fine-tune PhoBERT trên ViHSD ~3–5h nếu cần).
- [ ] Bảng so sánh: AI thuần vs rule engine vs hybrid trên từng mức nhiễu.

### 2.3. Công bố lexicon (Q1 2027)

- [ ] Chuẩn hóa schema lexicon (entry_id, term, labels, severity, provenance, version).
- [ ] Viết README + thống kê chất lượng (số mục từ, phân bố nhãn, nguồn dataset).
- [ ] Phát hành GitHub/HF kèm license tương thích CC BY-NC.

### 2.4. Phát hành (Q2 2027)

- [ ] Repo công khai: benchmark + lexicon + script sinh dữ liệu + bảng baseline.
- [ ] Báo cáo kỹ thuật ngắn (để trích dẫn trong bài nộp).

**Tiêu chí hoàn thành GĐ2:** người ngoài có thể tải về và tái lập bảng baseline
trong ≤1 giờ trên máy có GPU thông thường.

---

## 3. GIAI ĐOẠN 3 — Nộp bài benchmark cho IEEE SMC 2027 (CORE B)

### 3.1. Vì sao IEEE SMC 2027 (đã xác minh)

- **Hợp pháp & uy tín:** flagship conference của IEEE SMC Society (tổ chức từ 1971),
  được xác nhận trên ieeesmc.org ("SMC 2027 in Ho Chi Minh City, Vietnam").
- **CORE B** (theo công bố của chính IEEE SMC Society).
- **Địa điểm:** TP.HCM, 6–10/10/2027 (Sheraton Saigon Grand, đăng cai bởi HCM-UTE) —
  không tốn chi phí đi lại.
- **Deadline:** nộp bài ~15/04/2027, thông báo 30/06/2027, camera-ready 20/07/2027.
- **Index:** IEEE Xplore + Scopus.
- Chủ đề khớp khi đóng khung đúng: bài benchmark trình bày như *"resilience of
  intelligent moderation systems under evasion attacks + human-machine decision
  support (ALLOW/REVIEW/BLOCK)"* — khớp mảng computational intelligence /
  human-machine systems của SMC. KHÔNG đóng khung như bài NLP thuần.

### 3.2. Yêu cầu đặc thù của IEEE SMC (khác Springer)

- [ ] Format **IEEE 2 cột, ~6 trang** (kiểm tra CFP chính xác) — nén bài 12–15 trang
      Springer xuống còn bản cô đọng: chỉ giữ benchmark + kết quả chính, chi tiết
      đẩy xuống supplementary/artifact repo.
- [ ] **Chống self-plagiarism:** nếu bài ICCIES đã được chấp nhận/xuất bản, bài SMC
      PHẢI là bản mở rộng thực chất (benchmark mới, multi-seed, public baselines —
      chính là GĐ2), không được nộp lại nội dung cũ. Ghi rõ trong cover letter.
- [ ] Giải thích kỹ khái niệm NLP (macro-F1, ViHSD, label imbalance) vì reviewer SMC
      có thể không chuyên NLP.
- [ ] Đăng ký tài khoản IEEE + tuân thủ quy trình PDF eXpress nếu được yêu cầu.

### 3.3. Venue dự phòng (nếu SMC không phù hợp hoặc trượt)

| Venue | Rank | Deadline dự kiến | Ghi chú |
|---|---|---|---|
| **RANLP 2027** (Varna, Bulgaria) | CORE B | ~05/2027 | Có track resources/evaluation, cộng đồng thân thiện bài tài nguyên |
| **TSD 2027** (Springer LNAI) | CORE B | ~03/2027 | Rất nhận bài tài nguyên cho ngôn ngữ ít dữ liệu |
| **ADMA 2027** (Springer LNAI) | CORE B | ~05–06/2027 | Góc "applied data mining" |

- [ ] Kiểm tra xếp hạng mới nhất trên CORE portal (core.edu.au) trước mỗi lần nộp.

### 3.4. Viết bài benchmark

- [ ] Cấu trúc 6 trang: (1) benchmark + cách sinh, (2) lexicon resource, (3) baselines
      multi-seed + public models, (4) phân tích độ khó + label drift, (5) kết luận.
- [ ] Báo cáo trung thực kết quả âm tính (fusion không ý nghĩa ở đa số model...) —
      chấp nhận được với bài benchmark vì trọng tâm là tài nguyên.

### 3.5. Checklist rank B

- [ ] Artifact link công khai trong bài nộp
- [ ] ≥3 seeds cho hệ thống chính, mean ± std
- [ ] ≥2 model public (PhoBERT/ViSoBERT/ViHateT5) làm baseline
- [ ] Phân tích chất lượng benchmark (drift, difficulty, ceiling)
- [ ] Threats to validity đầy đủ
- [ ] Đúng format IEEE 6 trang + PDF eXpress

**Tiêu chí hoàn thành GĐ3:** bài nộp đúng deadline 15/04/2027 + artifact được trích dẫn trong bài.

---

## 4. GIAI ĐOẠN 4 — Journal Q2–Q3 (tùy chọn, 2027–2028)

- [ ] Mở rộng bài CORE B (≥40% nội dung mới): thêm dataset thứ hai ngoài ViHSD
      (VD: đánh giá trên ViHOS hoặc dữ liệu thu thập mới), thêm phân tích calibration,
      thêm ứng dụng thực tế (triển khai demo ALLOW/REVIEW/BLOCK).
- [ ] Ứng viên tạp chí: Applied Intelligence, PeerJ Computer Science,
      Social Network Analysis and Mining, hoặc tạp chí Scopus Q2/Q3 trong danh mục
      HĐGSNN phù hợp ngành CNTT.
- [ ] Lưu ý quy tắc chống self-plagiarism của Springer khi mở rộng từ proceedings.

---

## 5. Quản trị rủi ro

| Rủi ro | Mức | Biện pháp |
|---|---|---|
| Deadline ICCIES gần hơn dự kiến | Cao | Kiểm tra ngay hôm nay; nếu quá gần, cắt gọn GĐ1 còn các việc bắt buộc |
| Bài đã nộp ICEIR (dual submission) | Cao | Xác minh ngay; nếu đã nộp → dời toàn bộ nỗ lực sang GĐ2/GĐ3 |
| GPU 8GB không đủ multi-seed kịp | Trung bình | Giới hạn multi-seed cho 2 hệ thống; public models chỉ inference |
| Nén bài xuống 6 trang IEEE không đủ chỗ | Trung bình | Giữ nguyên tắc: bài nộp chỉ có benchmark + kết quả chính; chi tiết vào artifact repo |
| Self-plagiarism giữa bài ICCIES và bài SMC | Cao | Bài SMC bắt buộc là bản mở rộng thực chất (GĐ2); ghi rõ trong cover letter, không tái sử dụng text/bảng cũ |
| ViHSD license cấm phân phối lại | Trung bình | Phát hành code sinh benchmark thay vì dữ liệu biến đổi |
| Kết quả blind-spot không như kỳ vọng | Trung bình | Framing chuyển thành limitation + bài học cho cộng đồng (vẫn công bố được) |
| Chính sách AI-generated text của ICCIES | Cao | Viết lại bài bằng ngôn ngữ của tác giả trước camera-ready |
| Xếp hạng CORE thay đổi | Thấp | Kiểm tra core.edu.au trước mỗi lần nộp |

---

## 6. Các mốc quyết định (decision gates)

1. **Gate 1 — ngay hôm nay:** xác nhận deadline ICCIES + tình trạng ICEIR → chọn
   đường đi (nộp ICCIES hay chuyển thẳng GĐ2).
2. **Gate 2 — sau phân tích blind-spot:** RE bắt được ≥5–10% trong 504 mẫu → section
   mạnh; <5% → viết thành limitation.
3. **Gate 3 — 03/2027:** benchmark v1 + license + multi-seed xong → chốt bài nộp
   IEEE SMC 2027 (deadline 15/04/2027) và viết bài.
4. **Gate 4 — sau kết quả GĐ3:** accept → chuẩn bị GĐ4; reject → chỉnh sửa theo review
   và nộp venue dự phòng (RANLP/TSD/ADMA hoặc journal Q3).

---

## 7. File tham chiếu chính

| Nội dung | Đường dẫn |
|---|---|
| Đánh giá chi tiết bài hybrid + checklist cải thiện | `README_danh_gia_hybrid.md` |
| Kế hoạch rule engine gốc | `Rule_Engine_Plan.md` |
| Template bài nộp | `papers/main.tex` (llncs) |
| References hiện có | `papers/references.bib` |
| Kết quả fusion | `outputs/evaluation/fusion/comparison.md` |
| Rule engine tuned + robustness | `outputs/evaluation/rule_engine_tuned/` |
| Blind-spot (504 mẫu) | `outputs/evaluation/analysis/supplementary_report.md` |
| Ensemble (cần chạy lại) | scripts `31_ensemble_experiment.py` → `33_vote_fused_models.py` |
