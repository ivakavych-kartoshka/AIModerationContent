<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# evaluation/rule_ablations - ablation thành phần rule engine

> **Dùng để làm gì:** Mỗi biến thể tắt **một** thành phần của rule engine (`no_severity`, `no_confidence`, `no_context`, `no_views`, `no_fuzzy`) được chấm lại từ đầu: rule engine re-score validation + test, fusion re-tune alpha và threshold trên **validation**, đóng băng, rồi chấm test **một lần** (plan §44). Vì vậy mỗi biến thể có tham số riêng - câu hỏi là thành phần đó có đóng góp gì, không phải tham số nào tốt hơn. Kết quả: chỉ `confidence` là quan trọng (rule-only macro-F1 0.5258 -> 0.4914); các thành phần còn lại đều nằm trong nhiễu.

> **Sinh ra bởi:** `python scripts/50_rule_component_ablations.py`

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `full/` | Biến thể đầy đủ (đối chứng) - cũng là cấu hình đã ship, nên số ở đây trùng `evaluation/fusion/`. |
| `no_severity/` | Bỏ map severity (HATE/OFFENSIVE = 1.0): đổi 1.046/6.680 vector điểm rule nhưng hybrid không đổi quyết định nào. |
| `no_confidence/` | Đặt `entry_confidence = 1.0` cho mọi entry: thành phần duy nhất thật sự có đóng góp. |
| `no_context/` | Tắt cờ context-required. |
| `no_views/` | Tắt obfuscation views (chỉ exact + fuzzy): số comment bắn giảm 1.052 -> 779, precision/comment bắn 58.75% -> 69.58%. |
| `no_fuzzy/` | Tắt fuzzy match: không đổi gì vì fuzzy không khớp lần nào trên test. |
| `fusion/` | Thư mục fusion dùng tạm cho mỗi lần chạy (bị ghi đè giữa các biến thể; bản chốt nằm ở `results/`). |
| `results/<variant>/` | Ảnh chụp `metrics.json` + `fusion_predictions.csv` của từng biến thể - nguồn số cho bài báo. |
| `ablation_summary.json` | Bảng tổng hợp mọi biến thể + kiểm định paired bootstrap và McNemar của từng biến thể so với `full`. |
| `explain.md` | File này. |


## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `full` | thư mục | 40 file, 2.5 MB |
| `fusion` | thư mục | 7 file, 1.2 MB |
| `no_confidence` | thư mục | 41 file, 2.5 MB |
| `no_context` | thư mục | 40 file, 2.5 MB |
| `no_fuzzy` | thư mục | 40 file, 2.5 MB |
| `no_severity` | thư mục | 40 file, 2.4 MB |
| `no_views` | thư mục | 40 file, 2.4 MB |
| `results` | thư mục | 12 file, 7.2 MB |
| `ablation_summary.json` | file | 6.8 KB · 2026-11-01 12:45:54 |

## Cách đọc

1. Đọc `ablation_summary.json`: khoá `variants.<tên>.vs_full` là phần kiểm định - CI chứa 0 nghĩa là thành phần đó không đóng góp gì đo được.
2. `precision_on_fired` chỉ so được giữa các biến thể, không so với AI: nó là tỉ lệ comment bị rule bắn mà nhãn vàng không phải CLEAN.

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-11-01 13:13:04. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
