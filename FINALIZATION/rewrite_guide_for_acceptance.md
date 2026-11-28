# Hướng dẫn viết lại đoạn AI-condensed để đảm bảo được accept (ICCIES 2027)

Mục tiêu: thay thế văn bản do AI cô đọng bằng lời của tác giả, giữ nguyên toàn bộ số liệu/claim/cấu trúc, không vượt 15 trang, và tránh bị flag là "AI-generated text" theo chính sách ICCIES.

## 0. Nguyên tắc bất biến (KHÔNG được vi phạm)

1. **Không thêm/xoá số liệu.** Mọi con số, ngưỡng, p-value, khoảng tin cậy phải giữ nguyên. Nếu đổi một con số, phải có artifact chứng minh.
2. **Không thay đổi claim.** Nếu bài nói "gain is not distinguishable from zero (p=0.29)", không được viết lại thành "improves performance".
3. **Không thay đổi citation.** Mỗi `\cite{...}` phải giữ đúng vị trí ý nghĩa.
4. **Không vượt 15 trang.** Mỗi đoạn viết lại nên **ngắn bằng hoặc ngắn hơn** bản cô đọng hiện tại.
5. **Không dùng humanizer/paraphrase tool.** Đó vẫn là văn bản AI.
6. **Không chèn lỗi chính tả** để "trông giống người" — phản tác dụng, reviewer coi là thiếu chuyên nghiệp.

## 1. Dấu hiệu văn bản AI (tránh)

Reviewer và iThenticate có xu hướng bắt các mẫu sau — **tránh chúng**:

- Câu quá cân đối: "X does Y; Z does W." lặp lại nhiều lần.
- Cấu trúc "First, ... Second, ... Third, ..." quá đều.
- Từ nối chung chung lặp lại: "Furthermore", "Moreover", "Additionally", "In addition", "It is worth noting that".
- Kết luận đoạn kiểu "This highlights the importance of ...".
- Câu dài với nhiều mệnh đề phụ liên tiếp, không có câu ngắn xen kẽ.
- Danh sách (i)(ii)(iii) quá gọn và song song hoàn hảo.
- Tránh mọi câu bắt đầu bằng "We" liên tiếp quá 2 câu.
- Tránh cụm "It is important to note that", "It should be noted that", "It is worth mentioning".

## 2. Dấu hiệu văn bản người viết (nên có)

- **Câu ngắn xen kẽ câu dài.** Ít nhất 1 câu ≤ 12 từ trong mỗi đoạn.
- **Ví dụ cụ thể từ corpus** (đã mask): trích comment thật, ghi rõ nó thuộc correction hay regression nào.
- **Quan sát cá nhân**: "Chúng tôi nhận thấy...", "Điều bất ngờ là...", "Ban đầu chúng tôi nghĩ...".
- **Hedge tự nhiên**: "có vẻ", "trong trường hợp này", "theo kinh nghiệm của chúng tôi".
- **Chi tiết triển khai**: tên hàm, giá trị cụ thể, lý do chọn ngưỡng — những thứ AI thường bỏ qua.
- **Không đều**: đoạn này 3 câu, đoạn kia 6 câu; có đoạn chỉ 1 câu.

## 3. Quy trình viết lại từng đoạn (4 bước)

1. **Đọc bản gốc** (trong `_backup_precondense/`) để hiểu ý định ban đầu.
2. **Đọc bản cô đọng hiện tại** (trong `sections/`) để biết các số liệu đã giữ.
3. **Tắt máy/đóng file**, viết lại đoạn bằng lời của mình trên giấy hoặc editor trống, không nhìn bản cô đọng.
4. **So sánh**: kiểm tra mọi số liệu, citation, label có khớp không. Nếu thiếu số, thêm lại.

## 4. Phân bổ thời gian (2 ngày)

| Ngày | Việc | Mục tiêu |
|---|---|---|
| 15/11 sáng | P1: 12 đoạn heavy (discussion, results, analysis) | Viết lại hoàn toàn bằng tay |
| 15/11 chiều | P2: 11 đoạn moderate | Diễn đạt lại |
| 16/11 sáng | P3: 13 đoạn light + rà soát toàn bài | Chỉnh từ/câu |
| 16/11 chiều | Rebuild, kiểm tra trang/overfull/??, chạy test+audit | Xác nhận kỹ thuật |
| 17/11 sáng | Copy PDF, cập nhật checksum, upload CMT | Nộp trước 14:59 |

## 5. Mẹo theo từng section

- **Introduction**: đoạn "Two problems follow" và "Contributions" — viết lại danh sách đóng góp theo giọng của bạn, tránh song song hoàn hảo; thêm 1 câu về động lực cá nhân (tại sao chọn bài toán này).
- **Related work**: giữ citation, nhưng diễn đạt lại phần so sánh bằng lời của bạn; thêm nhận xét về hạn chế của từng công trình liên quan.
- **Data/Lexicon**: phần "Ethics of examples" và "lexicon is not a list of slurs" — rất dễ viết bằng lời của bạn vì là mô tả kỹ thuật cụ thể.
- **Method**: phần "Tuning" và "Evidence record" — thêm chi tiết triển khai (tên hàm, giá trị) mà AI đã lược bỏ.
- **Results**: "The hybrid does not improve accuracy" — viết lại kết quả theo giọng báo cáo thực nghiệm, không theo giọng tóm tắt.
- **Analysis**: "Case studies" — trích dẫn comment thật (đã mask) và giải thích cơ chế bằng lời của bạn.
- **Discussion**: "Why accuracy does not move" và "Design implications" — đây là phần **giá trị nhất của bài**, nên viết bằng giọng của bạn hoàn toàn; đây là nơi reviewer đánh giá "insight".
- **Conclusion**: viết lại ngắn gọn, không lặp abstract.

## 6. Checklist trước khi nộp (bắt buộc pass hết)

- [ ] Mọi con số trong bài khớp với `outputs/evaluation/...` (chạy `audit_iccies2027_assertions.py`).
- [ ] `python -m pytest FINALIZATION/tests -q` → 9 passed.
- [ ] `latexmk` build thành công, 0 overfull hbox, 0 undefined ref.
- [ ] Tổng số trang ≤ 15.
- [ ] Đọc lại toàn bài 1 lần bằng mắt người (không phải máy) để bắt câu AI-sounding.
- [ ] Kiểm tra metadata PDF rỗng (title/author/keywords) — đã OK.
- [ ] Font nhúng — đã OK.
- [ ] File nộp là `FINALIZATION/submission/iccies2027_final.pdf`, cập nhật checksum.
- [ ] Upload lên CMT trước 14:59 ngày 17/11/2026.

## 7. Điều tôi (AI) có thể giúp (an toàn)

- Kiểm tra **tính chính xác số liệu/claim** trong các đoạn bạn tự viết (factual check).
- So sánh bản bạn viết với bản gốc để đảm bảo không mất claim quan trọng.
- Gợi ý **cấu trúc** (không viết thay câu).

## Điều tôi KHÔNG nên giúp

- Viết thay câu/đoạn để "trông giống người" — đó vẫn là văn bản AI và vi phạm chính sách ICCIES.
- Paraphrase tự động để né detector.
- Thêm lỗi chính tả.

## Ghi chú cuối

Nếu không còn thời gian viết lại toàn bộ, ưu tiên **P1 (12 đoạn)** — đây là phần cô đọng nặng nhất và có rủi ro AI cao nhất. P2/P3 có thể giữ nguyên nếu cần. Nộp trước hạn vẫn quan trọng hơn; bài accept hay không phụ thuộc chủ yếu vào **ý tưởng và số liệu**, không phải văn phong.
