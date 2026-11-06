# Wortschatz Leipzig English Benchmark Report

## Dataset
- **Source:** Wortschatz Leipzig 2015 English News Corpus
- **Total sentences:** 29,934
- **Language:** English (news domain, 2015)
- **License:** CC-BY-NC-SA 3.0

## Splits
- **Train:** 14,967 (50.0%)
- **Validation:** 5,088 (17.0%)
- **Test:** 9,879 (33.0%)

## Statistics
### Overall
- Mean length: 121 chars
- Median length: 115 chars
- Mean words: 20.4

### By Split
- Train: 74 chars, 12.6 words
- Val: 130 chars, 21.8 words
- Test: 186 chars, 31.3 words

## Purpose
1. **Neutral English benchmark:** news text, no hate-speech labels (test OOD robustness)
2. **Cross-lingual evaluation:** compare VI (ViHSD hate-speech focused) vs EN (news, clean domain)
3. **Calibration & latency:** measure system behavior on long-tail English text
4. **Blind-spot analysis:** identify words/patterns that confuse multilingual backbone

## Quality Notes
- Stratified by length to ensure balanced distribution
- No preprocessing (raw news sentences, may contain URLs, entities, contractions)
- Ground truth unavailable; use backbone predictions as proxy
- News domain ≠ social media; expect lower hate-speech density
