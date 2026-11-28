# Response to Reviewers

**Paper ID:** 278  
**Paper Title:** SensitiveAI: An Ablation Study of Optimization Recipes for Vietnamese Hate Speech Detection  
**Track:** ICEIR 2026  
**Authors:** Pham Kha Vy, Cao Hong Nga  

---

We sincerely thank the Track Chairs and the three Reviewers for their constructive feedback, insightful comments, and unanimous recommendation of **Acceptance** for our manuscript. Below, we outline the revisions incorporated into the final camera-ready version of the paper and its supplementary material.

---

## 1. Summary of Major Revisions

### Multi-Seed Retraining Study (Addressing Reviewer #1, #2, and #3)
All three reviewers highlighted that our original submission reported results from a single random seed (seed 42), which left training variance unquantified.

**Action taken:**
1. We fully re-trained and evaluated all six system configurations across three independent random seeds (42, 43, 44) using strictly self-contained per-seed lineages (so chained checkpoints and stages never mix seeds).
2. We added **Section 4.2 (*Stability Across Training Seeds*)** and **Table 3 (*multi-seed ablation*)** directly to the main paper, reporting the test metrics as $\text{mean} \pm \text{standard deviation}$ for Accuracy, Macro-F1, HATE-class F1, and Macro ROC-AUC.
3. The empirical results across seeds show that Macro-F1 standard deviations are all below $0.01$ (ranging from $0.0016$ to $0.0097$). This confirms that while training variance exists, it is substantially smaller than sampling noise and does not alter the paper's core conclusion: no single optimization recipe achieves a statistically distinct separation over the others on this benchmark.
4. We updated the **Abstract**, **Methodology (Section 3.1)**, **Threats to Validity (Section 5.2)**, and **Conclusion (Section 6)** to explicitly discuss the multi-seed findings.

### Synchronizing Supplementary Material
We updated **Section 1** and **Section 8 (*Complementary Experiments and Seed Variance*)** of the supplementary material (`papers/ICEIR2026/supplementary/main.pdf`) to reconcile the multi-seed execution, noting that deep diagnostics (calibration, error-overlap matrices, stratified subsets) are reported on the primary frozen run (seed 42), while training stability is verified via Table 3 in the main paper.

### Public Code Repository (De-anonymization)
In accordance with camera-ready guidelines, we updated the Data and Code Availability section to link to our permanent, public GitHub repository:  
`https://github.com/ivakavych-kartoshka/AIModerationContent`

---

## 2. Point-by-Point Responses to Reviewers

### Reviewer #1
> **Comment:** *"The authors present SensitiveAI, a controlled ablation study comparing six system configurations that share the same data split, label mapping, backbone (microsoft/mdeberta-v3-base), and random seed, differing only in their optimization strategies... Overall, the study is clearly structured; however, the reliance on a single random seed should be acknowledged as a limitation."*

**Response:**
We thank the reviewer for the positive assessment and recommendation of acceptance. As suggested, we have expanded our study to address the single-seed limitation directly. In Section 4.2 and Table 3, we now report results across three independent seeds (42, 43, 44). Furthermore, Section 5.2 (*Threats to Validity*) and Section 6 (*Future Work*) explicitly discuss run-to-run training variance alongside bootstrap sampling uncertainty.

---

### Reviewer #2
> **Comment:** *"This paper presents a carefully controlled ablation study of optimization recipes for Vietnamese hate speech detection using a fixed mDeBERTa-v3 backbone and the UIT-ViHSD dataset. The main strength of the paper is its methodological rigor... The paper is also commendable for openly reporting negative findings...*  
> *The main limitation is the relatively limited methodological novelty... In addition, all experiments are conducted using a single random seed... Furthermore, the best-performing system does not exceed the strongest reported Vietnamese-specific models in the literature...*  
> *However, the following improvement is strongly recommended before final acceptance: Repeat all experiments using multiple random seeds and report the mean ± standard deviation. This is necessary to verify the stability and statistical reliability of the conclusions."*

**Response:**
We thank Reviewer #2 for recognizing the methodological rigor, duplicate leakage safeguards, and the scientific value of reporting negative results.
1. **Multi-seed Retraining:** As strongly recommended, we repeated all experiments across three independent random seeds (42, 43, 44) with self-contained per-seed lineages. As reported in Table 3:
   - Baseline M1: $\text{Acc} = 0.8741 \pm 0.0048$, $\text{Macro-F1} = 0.6580 \pm 0.0097$
   - Custom-head M2: $\text{Acc} = 0.8760 \pm 0.0012$, $\text{Macro-F1} = 0.6608 \pm 0.0037$
   - Chained DLR M3: $\text{Acc} = 0.8717 \pm 0.0007$, $\text{Macro-F1} = 0.6632 \pm 0.0016$
   - Curriculum M4: $\text{Acc} = 0.8732 \pm 0.0031$, $\text{Macro-F1} = 0.6652 \pm 0.0058$
   - Extended probe M5: $\text{Acc} = 0.8708 \pm 0.0065$, $\text{Macro-F1} = 0.6574 \pm 0.0038$
   - Focal probe M6: $\text{Acc} = 0.8659 \pm 0.0031$, $\text{Macro-F1} = 0.6611 \pm 0.0022$  
   The narrow standard deviations confirm that training variance does not separate the recipes.
2. **Comparison with Vietnamese-Specific Models:** In Section 4.7 (Table 4) and Section 5.3, we explicitly contextualize our results against published models like PhoBERT and ViSoBERT, highlighting that while specialized pre-training on domain text (social media) remains superior, our controlled study proves that post-hoc optimization tricks on multilingual backbones yield only marginal, statistically indistinguishable differences.

---

### Reviewer #3
> **Comment:** *"The paper presents SensitiveAI, a controlled ablation study of optimization strategies for Vietnamese hate speech detection using a fixed mDeBERTa-v3 backbone and UIT-ViHSD. Its main strength is experimental rigor... More importantly, all experiments use one random seed. The reported uncertainty therefore does not establish stability across retraining, limiting conclusions about the reliability of differences between optimization strategies. Acceptance with minor revision is supported, with training-seed variability remaining the principal unresolved limitation."*

**Response:**
We are grateful to Reviewer #3 for the thoughtful review and support for acceptance. We completely agree that establishing stability across retraining is critical. By retraining all systems across seeds 42, 43, and 44 and providing Table 3, we demonstrate that retraining stability holds (Macro-F1 std < 0.01 across all recipes). We explicitly state in Section 4.2 that the ranking shifts slightly across seeds (e.g., M4 mean Macro-F1 edging M3 by 0.002), further reinforcing our central message: complex optimization tricks do not reliably improve upon simpler recipes under retraining variance.

---

*The authors have thoroughly proofread the entire manuscript to ensure strict compliance with Springer LNNS formatting and the 12–15 page limit.*
