# Train embedding (fastText) + trong so thien class

> Corpus: ViHSD + ViCM + Bontori (133883 dong da tokenize bang underthesea). fastText skipgram dim=100, epochs=8. Centroid moi class = trung binh vector token trong class do (trong so DF).

## Phan bo term ung vien theo label du doan

| Label | Tat ca ung vien | Sau filter |
|---|---|---|
| HATE | 967 | 967 |
| OFFENSIVE | 240 | 240 |
| CLEAN | 248 | 0 |
| **Tong** | **1455** | **1207** |

## Vi du term nghieng ve HATE (margin cao nhat)

### HATE (967 term)

| term | cos_hate | cos_off | cos_clean | margin |
|---|---|---|---|---|
| `assmin` | 0.4945 | 0.4422 | 0.4300 | +0.0523 |
| `lũ chó` | 0.4392 | 0.3897 | 0.3900 | +0.0491 |
| `chó` | 0.5121 | 0.4634 | 0.4337 | +0.0487 |
| `óc chó` | 0.1665 | 0.1182 | 0.0961 | +0.0483 |
| `bọn chó` | 0.4071 | 0.3622 | 0.3218 | +0.0449 |
| `bake` | 0.4957 | 0.4510 | 0.4237 | +0.0447 |
| `ăn thịt chó` | 0.2510 | 0.2067 | 0.1909 | +0.0444 |
| `đmm` | 0.5238 | 0.4797 | 0.4470 | +0.0442 |
| `lũ ngu` | 0.5749 | 0.5311 | 0.4897 | +0.0438 |
| `ra mày` | 0.5452 | 0.5019 | 0.4646 | +0.0433 |
| `m đéo` | 0.5953 | 0.5522 | 0.5081 | +0.0431 |
| `mày ngu` | 0.5195 | 0.4767 | 0.4223 | +0.0428 |
| `mày tao` | 0.6569 | 0.6142 | 0.5791 | +0.0427 |
| `mày đã` | 0.5140 | 0.4713 | 0.4323 | +0.0427 |
| `mày đéo` | 0.6530 | 0.6110 | 0.5696 | +0.0420 |

### OFFENSIVE (240 term)

| term | cos_hate | cos_off | cos_clean | margin |
|---|---|---|---|---|
| `m quá` | 0.2431 | 0.2846 | 0.2391 | +0.0415 |
| `dạy quá` | 0.4842 | 0.5173 | 0.4780 | +0.0332 |
| `cắp content` | 0.4851 | 0.5153 | 0.4791 | +0.0302 |
| `ăn cắp content` | 0.4514 | 0.4775 | 0.4388 | +0.0261 |
| `mất dạy quá` | 0.4763 | 0.5024 | 0.4730 | +0.0260 |
| `đỉ` | 0.3979 | 0.4237 | 0.3877 | +0.0258 |
| `tỏ` | 0.4163 | 0.4421 | 0.4125 | +0.0258 |
| `đúng như` | 0.4378 | 0.4632 | 0.4220 | +0.0254 |
| `nhìn nhận` | 0.5607 | 0.5937 | 0.5693 | +0.0244 |
| `tốt nhất là` | 0.4860 | 0.5193 | 0.4963 | +0.0230 |
| `không công nhận` | 0.5340 | 0.5569 | 0.5292 | +0.0229 |
| `có cách` | 0.6024 | 0.6265 | 0.6036 | +0.0229 |
| `content` | 0.4634 | 0.4860 | 0.4486 | +0.0226 |
| `đọc hiểu` | 0.5387 | 0.5610 | 0.5280 | +0.0223 |
| `xl` | 0.6164 | 0.6385 | 0.6125 | +0.0221 |

## Cach doc

- `weight_*` = phan phoi mem: term nay 'thien' ve class nao theo ngu canh hoc duoc.
- `class_margin` = do ro rang; margin nho = term trung tinh, khong nen giu.
- Day moi la buoc **cham diem**; buoc do F1 tren validation se lam o script 38.
- Chua dung vao lexicon chinh thuc (plan section 44).
