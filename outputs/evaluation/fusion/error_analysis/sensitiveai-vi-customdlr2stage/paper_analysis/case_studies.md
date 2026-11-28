# Case Studies - AI Wrong → Fusion Right (sensitiveai-vi-customdlr2stage)

These examples demonstrate the explainability advantage of the hybrid approach.

## Case Study 1

**Index:** 5311  
**Text:** `chảm`  

| Item | Prediction |
|---|---|
| Ground truth | CLEAN |
| AI-only prediction | OFFENSIVE ❌ |
| Fusion prediction | CLEAN ✅ |

**Rule engine scores:** rule_hate=0.0000, rule_offensive=0.0000

**Explanation:** Lexical rule signals provide interpretable evidence supporting the corrected label.

---

## Case Study 2

**Index:** 5886  
**Text:** `dành cho lũ quan ngại`  

| Item | Prediction |
|---|---|
| Ground truth | HATE |
| AI-only prediction | OFFENSIVE ❌ |
| Fusion prediction | HATE ✅ |

**Rule engine scores:** rule_hate=0.4976, rule_offensive=0.0000

**Explanation:** Lexical rule signals provide interpretable evidence supporting the corrected label.

---

## Case Study 3

**Index:** 6366  
**Text:** `quá chán`  

| Item | Prediction |
|---|---|
| Ground truth | CLEAN |
| AI-only prediction | OFFENSIVE ❌ |
| Fusion prediction | CLEAN ✅ |

**Rule engine scores:** rule_hate=0.0000, rule_offensive=0.0000

**Explanation:** Lexical rule signals provide interpretable evidence supporting the corrected label.

---

