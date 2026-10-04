<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# scripts - các entry point

> **Dùng để làm gì:** 5 script chạy theo thứ tự tăng dần: chuẩn bị dữ liệu -> kiểm tra môi trường -> train -> đánh giá -> so sánh -> sinh tài liệu folder. Mỗi script chỉ là lớp vỏ mỏng gọi hàm trong `src/`.

> **Sinh ra bởi:** Chỉnh tay.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `00_prepare_data.py` | Tải + làm sạch + tách split ViHSD, kiểm tra SHA-256. Chạy 1 lần đầu tiên. |
| `01_train_all.py` | Train lần lượt cả 4 model. Mỗi model một `run_XXX` riêng. |
| `02_evaluate_all.py` | Chấm test cho các run đã train, tune threshold trên validation, đo tốc độ. |
| `03_compare_models.py` | Gộp kết quả 4 model thành bảng CSV/Markdown, biểu đồ, và file LaTeX cho paper. |
| `04_check_setup.py` | Kiểm tra môi trường, split, config, và forward pass. **Không** train. |
| `05_document_folders.py` | Sinh lại `explain.md` cho toàn bộ cây thư mục + bảng liệt kê các run. |
| `README.md` | Hướng dẫn dùng từng script. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `00_prepare_data.py` | file | 520 B · 2026-10-04 10:46:09 |
| `01_train_all.py` | file | 4.6 KB · 2026-10-04 11:40:09 |
| `02_evaluate_all.py` | file | 2.6 KB · 2026-10-04 11:25:20 |
| `03_compare_models.py` | file | 673 B · 2026-10-04 11:39:41 |
| `04_check_setup.py` | file | 4.5 KB · 2026-10-04 11:39:41 |
| `05_document_folders.py` | file | 3.2 KB · 2026-10-04 12:36:53 |
| `explain.md` | file | 2.3 KB · 2026-10-04 12:58:19 |
| `README.md` | file | 2.2 KB · 2026-10-04 11:31:08 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._
