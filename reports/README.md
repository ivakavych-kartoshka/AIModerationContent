# Báo cáo: Mô hình xác suất Naive Bayes cho lexicon của rule engine

Thư mục này chứa một **báo cáo ngắn tiếng Việt** (LaTeX) giải thích cho mentor:

- Kiến trúc toàn hệ thống (nhánh AI mDeBERTa + nhánh rule engine → fusion).
- Mô hình xác suất Naive Bayes sinh lexicon (công thức log-odds + softmax + ngưỡng multi-label).
- Kết quả: rule engine **có** cải thiện có ý nghĩa (`+0.0245`, `p=0.037`) nhưng fusion **không** (`+0.0030`, `p=0.351`).
- Toàn bộ chuỗi 4 hướng đã thử + các hình minh hoạ.

## Tệp

| Tệp | Nội dung |
|---|---|
| `bao_cao_mo_hinh_xac_suat.tex` | Nguồn LaTeX của báo cáo (tiếng Việt) |
| `bao_cao_mo_hinh_xac_suat.pdf` | **PDF đã biên dịch sẵn** (9 trang, XeLaTeX) |
| `figures/system_architecture.png` | Kiến trúc toàn hệ thống (copy từ `papers/figures/`) |
| `figures/directions_rule_vs_fusion.png` | Nghịch lý: rule engine tăng vs fusion nhích |
| `figures/directions_delta_ci.png` | Forest plot Δ + CI 95% mọi hướng |
| `figures/directions_evolution.png` | macro-F1 rule engine qua từng hướng |
| `figures/directions_per_class.png` | Heatmap F1 từng lớp → nút thắt ở OFFENSIVE |
| `figures/lexicon_macro_f1.png`, `lexicon_per_class_f1.png` | Hình phụ về các biến thể lexicon |

> **Lưu ý:** các hình trong `figures/` là **bản copy**. Muốn cập nhật, chạy
> `python scripts/44_plot_improvement_directions.py` (ghi vào
> `outputs/lexicon_experiments/figures/`) rồi copy lại, **không sửa tay**.
>
> `system_architecture.png` do `python scripts/30_plot_system_architecture.py` sinh.

## Biên dịch

Khuyến nghị **XeLaTeX** (hỗ trợ tiếng Việt tốt nhất, dùng `fontspec` + `polyglossia`):

```bash
cd reports
xelatex bao_cao_mo_hinh_xac_suat.tex
xelatex bao_cao_mo_hinh_xac_suat.tex   # chạy lần 2 để ổn định tham chiếu/hình
```

> ✅ Đã kiểm chứng: biên dịch thành công bằng XeLaTeX (MiKTeX 25.12) → **PDF 9
> trang**, không còn cảnh báo tham chiếu chưa định nghĩa. Bản PDF đã build sẵn ở
> `bao_cao_mo_hinh_xac_suat.pdf`.
>
> Trên máy này MiKTeX nằm ở
> `C:\Users\Lenovo\AppData\Local\Programs\MiKTeX\miktex\bin\x64`
> (thư mục này chưa có trong `PATH`). Nếu `xelatex` không tìm thấy, chạy:

```powershell
$env:PATH += ";C:\Users\Lenovo\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
```

Nếu máy không có font **Times New Roman / Arial / Consolas**, mở `.tex` và đổi:

```latex
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}
```

### Phương án dự phòng với `pdflatex`

Thay khối đầu file:

```latex
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{vietnamese}
\setotherlanguage{english}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
```

bằng:

```latex
\usepackage[utf8]{inputenc}
\usepackage[T5]{fontenc}
\usepackage[vietnamese]{babel}
```

rồi biên dịch:

```bash
pdflatex bao_cao_mo_hinh_xac_suat.tex
pdflatex bao_cao_mo_hinh_xac_suat.tex
```

## Nguồn số liệu

Mọi con số trong báo cáo được **sao chép nguyên văn** từ artifact do các script
40–44 sinh ra (xem Appendix trong báo cáo). Không có số nào tính lại bằng tay,
và **số liệu SOTA cũ không bị ghi đè** (mọi artifact mới nằm trong thư mục riêng
`outputs/evaluation/rule_engine_nb/` và `outputs/evaluation/fusion_nb/`).
