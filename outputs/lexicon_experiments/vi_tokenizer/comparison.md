# Tách lexicon từ ViHSD / ViCM / bontori — so sánh tokenizer tiếng Việt

> Ngưỡng: min_toxic_df=5, max_clean_df=1, TF-IDF gate mặc định; splits=train,dev,test. Che do `word` = underthesea word_tokenize (từ ghép → 1 token, có trigram). Che do `syllable` = cách hiện tại (split khoảng trắng, chỉ unigram+bigram).

## 1. So luong term moi tim duoc

| Nguồn | `syllable` (hiện tại) | `word` (underthesea) | Chỉ có ở `word` | Chỉ có ở `syllable` | Trùng nhau |
|---|---|---|---|---|---|
| vihsd | 228 | 192 | 38 | 74 | 154 |
| vicm | 647 | 542 | 62 | 167 | 480 |
| bontori | 77 | 86 | 38 | 29 | 48 |
| **Tổng** | **952** | **820** | **138** | **270** | **682** |

## 2. Ví dụ term mà CHỈ chế độ `word` (tokenizer tiếng Việt) tìm được

### vihsd — 38 term

```
banh xac   bưng   bắn chết   cat   chùi   chùi lol   chất độc   con chó này   các vị   của chúng   dm thằng   duma   dân trí thấp lại   giải tỏa   hiếp   họng   im mẹ mồm   lam gi   là luật   làm cán bộ   làm nô lệ   me no   mặt con   mẹ cái   mị dân
```

### vicm — 62 term

```
bi   chiến tranh du kích   chó chết   chật   chứng tỏ   con dân   các nước   cũng chả   cụ   dickhead   dẹp   dở   gay   go away   hèn   hại   hồn   khi stream chán vl   khác đi   khôn   kìa   les   loại ngu   làm như   lông
```

### bontori — 38 term

```
bao giờ thống nhất   bắn chết   bị mỹ   bố con nhà này   cho mỹ   chính quyền tay sai   chết thì được   có cl   của ông bố   diệm có   diệm thì   dân hàn   ghét trấn thành   gia đình này nó   là tay sai của   làm cách mạng màu   làm tay sai cho   lệ con   mấy đứa fan   ngáo quyền lực   ngụy quyền   nhận mình là   npc   quân tử trả thù   quân tử trả thù năm
```

## 3. Ví dụ term mà CHỈ chế độ `syllable` (hiện tại) tìm được

### vihsd — 74 term

```
boác   bò này   bệnh nặng   bệnh tâm   bọn quan   bọn trung   bợ đít   chai sting vô   chóp   chúng bây   con bò đỏ   con cac   cut   cái chế   cái con này   cái thể   cây đinh   công cộng   của loài   dâm loạn viên   dân để   gì chúng   họ nhà   khùng quá   ly chứ
```

### vicm — 167 term

```
acc   ai quan   argue   biết chửi   bà già   bê đê bóng   bị ăn   cay dái   chiến tranh du   cho bố   chuyện như   chịu tìm   chửi là đúng   chửi đổng   compare   con nhà   con súc   càng ngày   cái xàm   cái đất   còn ko   có lý   có nhà   có tí   cùn
```

### bontori — 29 term

```
bao giờ thống   bị ngu à   bọn đầu   cháo đá bát   chính quyền tay   cộng sản thì   dân hàn quốc   giấu đầu hở   giờ thống nhất   không có trí   là trung lập   nhức đau khổ   nể mấy đứa   quan lệch lạc   quyền tay sai   quân tử trả   sinh đẻ là   tam quan lệch   thù năm chưa   tiên thống nhất   triều tiên thống   tử trả thù   vớ va vớ   áp tôn giáo   đau nhức đau
```

## 4. Cách đọc

- `word` nhiều term hơn hẳn = tokenizer tiếng Việt tách từ ghép đúng → tìm được term dài (trigram) mà `split` bỏ sót.
- Term ở cột 'chỉ có ở word' thường là cụm 2-3 từ ghép thật (vd `khốn_nạn` + ngữ cảnh).
- Term ở cột 'chỉ có ở syllable' thường là mảnh âm tiết rời không mang nghĩa.
- **Chưa** đưa các term này vào lexicon chính thức — đây là nhánh thí nghiệm để so sánh.
