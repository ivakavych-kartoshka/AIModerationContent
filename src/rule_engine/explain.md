<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# rule_engine - rule engine từ vựng

> **Dùng để làm gì:** Bộ phát hiện từ vựng **đứng độc lập** với model AI (plan §41): chuẩn hoá văn bản nhưng giữ nguyên offset về câu gốc, dựng view đối chiếu chống ngụy trang, khớp từ điển đã gán nhãn và cho ra điểm `rule_hate_score` / `rule_offensive_score` kèm bằng chứng giải thích được (span, từ điển, loại khớp, hệ số tin cậy). Đây là đầu vào cho bước fusion AI + rule ở giai đoạn sau (plan §40).

> **Sinh ra bởi:** Không sinh ra - mã nguồn. Kết quả đánh giá nằm ở `evaluation/rule_engine/`.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `config.py` | `NormalizeConfig` / `ObfuscationConfig` / `MatcherConfig` / `ScoringConfig` / `RuleEngineConfig`, nạp từ `configs/rule_engine.yaml` và kiểm tra hợp lệ. |
| `normalize.py` | Chuẩn hoá NFC + hạ chữ + bỏ ký tự vô hình + gộp space (kèm giới hạn lặp ký tự), **giữ offset map** về câu gốc; không xóa dấu tiếng Việt (plan §37). |
| `obfuscation.py` | Dựng view `base` / `plain` (không dấu + leet + gộp lặp) / `compact` (bỏ dấu cách, dấu câu) và phân loại biến đổi thành `match_type` + `transformation_trace` (plan §38). |
| `lexical_classifier.py` | Nạp `verified_lexicon.csv` thành `Lexicon` (chỉ dùng entry có `review_status` được cho phép) với tra cứu từ nhanh. |
| `matcher.py` | Khớp `exact` / `accentless_alias` / `leet` / `repeat_collapsed` / `separator_inserted` / `obfuscated_combined` / `fuzzy` (Levenshtein có chặn), khử trùng span và ghi đầy đủ bằng chứng. |
| `scoring.py` | Điểm `severity x confidence x match_factor x context_factor` (phủ định, trích dẫn, `context_required`) rồi gộp mỗi nhãn bằng `1 - prod(1 - s_i)` (plan §39). |
| `engine.py` | `RuleEngine.analyze()` gom toàn bộ pipeline và trả về evidence dict; `from_config` / `from_lexicon` để dựng nhanh. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 8 file, 74.7 KB |
| `__init__.py` | file | 1.4 KB · 2026-10-17 01:00:08 |
| `config.py` | file | 5.9 KB · 2026-10-17 01:00:08 |
| `engine.py` | file | 4.3 KB · 2026-10-17 01:00:08 |
| `explain.md` | file | 3.3 KB · 2026-10-19 15:06:20 |
| `lexical_classifier.py` | file | 4.6 KB · 2026-10-17 01:00:08 |
| `matcher.py` | file | 13.5 KB · 2026-10-17 01:00:08 |
| `normalize.py` | file | 5.4 KB · 2026-10-17 01:00:08 |
| `obfuscation.py` | file | 9.0 KB · 2026-10-17 01:00:08 |
| `scoring.py` | file | 6.3 KB · 2026-10-17 01:00:08 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-19 15:09:39. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
