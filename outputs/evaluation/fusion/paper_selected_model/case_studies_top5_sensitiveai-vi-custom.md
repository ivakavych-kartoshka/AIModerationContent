# Case Studies (AI-only wrong -> Fusion correct) - sensitiveai-vi-custom

## Case 1

**Text:** lý khùng thì có,kk

| Item | Prediction |
|---|---|
| Ground Truth | CLEAN |
| AI-only | HATE (WRONG) |
| Fusion (hybrid) | CLEAN (CORRECT) |

**Rule evidence:** rule_hate=0.0000, rule_offensive=0.0000

---

## Case 2

**Text:** hóng bro tét đít thằng tùng

| Item | Prediction |
|---|---|
| Ground Truth | CLEAN |
| AI-only | HATE (WRONG) |
| Fusion (hybrid) | CLEAN (CORRECT) |

**Rule evidence:** rule_hate=0.6046, rule_offensive=0.0000

---

## Case 3

**Text:** sạo loxz quền của chị 3

| Item | Prediction |
|---|---|
| Ground Truth | HATE |
| AI-only | CLEAN (WRONG) |
| Fusion (hybrid) | HATE (CORRECT) |

**Rule evidence:** rule_hate=0.6268, rule_offensive=0.0000

---

## Case 4

**Text:** tâm nguyễn bỏ mớ tiền ra thuê hẳn một cái máy bay riêng, đi 1 mình thì lây ai, lây cho con mẹ bạn à? thông tin trên google sợt là ra cũng đéo chịu tìm hiểu, vào đây chửi adua phong trào. \"đạo đức rách\" mà cứ ngồi đấy mà nói đạo lý.

| Item | Prediction |
|---|---|
| Ground Truth | HATE |
| AI-only | OFFENSIVE (WRONG) |
| Fusion (hybrid) | HATE (CORRECT) |

**Rule evidence:** rule_hate=0.9990, rule_offensive=0.0000

---

## Case 5

**Text:** lủ chó

| Item | Prediction |
|---|---|
| Ground Truth | HATE |
| AI-only | OFFENSIVE (WRONG) |
| Fusion (hybrid) | HATE (CORRECT) |

**Rule evidence:** rule_hate=0.9882, rule_offensive=0.0000

---

