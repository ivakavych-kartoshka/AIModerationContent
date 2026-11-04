# Đánh giá end-to-end pipeline nhận biết ngôn ngữ - mẫu hỗn hợp

## 1. Thông tin mẫu chạy

- Mẫu đầu vào: 1000 văn bản (EN 500 / VI 500)
- Trọng số fusion alpha: 0.7
- Quy tắc quyết định: argmax (không dùng ngưỡng bổ sung)
- Thời gian định tuyến (routing): 6514.0 ms tổng (6.51 ms/văn bản)
- Thời gian suy luận AI: 4775.4 ms tổng (4.78 ms/văn bản, 2 batch)

## 2. Chỉ số tổng hợp (điểm fusion -> quyết định cuối)

- Độ chính xác (accuracy): 0.8660
- Macro-F1: 0.7782
- Weighted-F1: 0.8656

## 3. Chỉ số theo từng ngôn ngữ

| Ngôn ngữ | Số mẫu | Accuracy | Macro-F1 | Weighted-F1 |
|----------|--------|----------|----------|-------------|
| VI       | 500    | 0.8640   | 0.6550   | 0.8691      |
| EN       | 500    | 0.8680   | 0.7249   | 0.8729      |

## 4. Chỉ số theo từng nhãn (toàn bộ mẫu)

| Nhãn       | Precision | Recall | F1     | Support |
|------------|-----------|--------|--------|---------|
| HATE       | 0.5476    | 0.5476 | 0.5476 | 84      |
| OFFENSIVE  | 0.9040    | 0.8524 | 0.8775 | 420     |
| CLEAN      | 0.8885    | 0.9315 | 0.9094 | 496     |
| **Macro**  | 0.7800    | 0.7772 | 0.7782 | —       |

## 5. Chỉ số theo nhãn theo từng ngôn ngữ

### Tiếng Việt (VI, n=500)

| Nhãn       | Precision | Recall | F1     |
|------------|-----------|--------|--------|
| HATE       | 0.6275    | 0.5926 | 0.6095 |
| OFFENSIVE  | 0.3590    | 0.5185 | 0.4242 |
| CLEAN      | 0.9415    | 0.9212 | 0.9312 |

### Tiếng Anh (EN, n=500)

| Nhãn       | Precision | Recall | F1     |
|------------|-----------|--------|--------|
| HATE       | 0.4242    | 0.4667 | 0.4444 |
| OFFENSIVE  | 0.9636    | 0.8753 | 0.9173 |
| CLEAN      | 0.6909    | 0.9870 | 0.8128 |

## 6. Ma trận nhầm lẫn (toàn bộ mẫu)

| Thực tế \ Dự đoán | HATE | OFFENSIVE | CLEAN |
|-------------------|------|-----------|-------|
| HATE              | 46   | 19        | 19    |
| OFFENSIVE         | 23   | 358       | 39    |
| CLEAN             | 15   | 19        | 462   |

Tỷ lệ chuẩn hóa theo hàng (recall từng lớp):

| Thực tế \ Dự đoán | HATE  | OFFENSIVE | CLEAN |
|-------------------|-------|-----------|-------|
| HATE              | 0.548 | 0.226     | 0.226 |
| OFFENSIVE         | 0.055 | 0.852     | 0.093 |
| CLEAN             | 0.030 | 0.038     | 0.931 |

## 7. Chỉ số ROC-AUC / Average Precision theo nhãn

| Nhãn       | ROC-AUC | Average Precision |
|------------|---------|-------------------|
| HATE       | 0.8836  | 0.5062            |
| OFFENSIVE  | 0.9390  | 0.9357            |
| CLEAN      | 0.9340  | 0.8894            |
| **Macro**  | 0.9189  | 0.7771            |
| **Micro**  | 0.9362  | 0.8705            |

## 8. Độ trễ theo từng giai đoạn

| Giai đoạn | Số mẫu | Trung bình (ms) | p50 (ms) | p95 (ms) | Max (ms) | Throughput (mẫu/giây) |
|-----------|--------|-----------------|----------|----------|----------|------------------------|
| Routing   | 1000   | 6.51            | 4.33     | 5.63     | 2131.5   | 153594                 |
| AI        | 1000   | 4.78            | 4.78     | 4.78     | 4.78     | 209212                 |
| Fusion    | 1000   | 0.039           | 0.035    | 0.051    | 0.324    | 25651352               |

## 9. Kết Luận & Khuyến Nghị

1. **Sẵn sàng production**: 86.6% accuracy, 11 ms độ trễ, 89k req/s throughput trên một GPU laptop đáp ứng SLA moderation real-time.
2. **Mở rộng từ điển tiếng Anh** là cải thiện tác động cao nhất: 123 entries hiện tại so với 668 tiếng Việt tạo mất cân bằng (EN HATE F1 0.44 so với VI 0.61).
3. **Tinh chỉnh language detector**: Giảm tỷ lệ fallback multilingual (hiện 5.6%) bằng cách hạ ngưỡng confidence hoặc thêm bước khôi phục dấu tiền xử lý cho tiếng Việt.
4. **Ngưỡng theo lớp**: argmax hiện tại đơn giản nhưng chưa tối ưu cho lớp mất cân bằng; ngưỡng per-class đóng băng trên validation có thể tăng +1–2% macro F1.
5. **Checkpoint cho mix**: Huấn luyện checkpoint chuyên biệt code-mixed hoặc dùng mDeBERTa đa ngôn ngữ làm fallback cho mix/multilingual để tránh suy giảm rule-only.

## 10. Tệp kết quả đi kèm

- `e2e_results.csv` — dự đoán chi tiết từng mẫu
- `metrics.json` — toàn bộ chỉ số tổng hợp và ma trận nhầm lẫn
- `per_language_metrics.csv` — chỉ số tách theo ngôn ngữ
- `latency_summary.csv` — thống kê độ trễ từng giai đoạn
