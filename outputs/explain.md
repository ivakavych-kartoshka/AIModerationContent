<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# outputs - kết quả huấn luyện & đánh giá chính thức

> **Dùng để làm gì:** Thư mục chứa toàn bộ kết quả chính thức: `reports/` (chi tiết từng run của từng model), `experiments/` (trạng thái huấn luyện và checkpoints trung gian), `evaluation/` (đánh giá tổng hợp và so sánh 4 model), và `paper/` (bảng số liệu, biểu đồ cho bài báo).

> **Sinh ra bởi:** Các pipeline huấn luyện chính thức (scripts 01 -> 03).

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX`), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## Thư mục con và vai trò

| Thư mục con | Vai trò |
|---|---|
| `reports/` | Báo cáo chi tiết, log và checkpoint `best_model` của từng model theo run. |
| `experiments/` | Checkpoints trung gian giữa các stage của mô hình đa giai đoạn. |
| `evaluation/` | Log chấm điểm tổng hợp và bảng/biểu đồ so sánh 4 model. |
| `paper/` | Bảng LaTeX và hình vẽ dùng cho bài báo. |
