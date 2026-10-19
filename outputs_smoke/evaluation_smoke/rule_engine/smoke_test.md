# Rule engine smoke test

- generated_at: 2026-10-19T14:29:26
- result: PASSED
- checks: 24 (0 failed)
- config: configs\rule_engine.yaml
- lexicon: 668 entries (data/processed/lexicon/verified_lexicon.csv)
- split: validation (limit=60)

## Checks

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | config loads + validates | OK | configs\rule_engine.yaml |
| 2 | lexicon loads | OK | 668 entries |
| 3 | engine builds | OK | matcher ready |
| 4 | keeps Vietnamese diacritics | OK | 'thằng mất dạy, tao ghét nó' |
| 5 | lowercases | OK |  |
| 6 | collapses whitespace runs | OK | 'thằng mất dạy, tao ghét nó' |
| 7 | offset map length == normalized length | OK | 26 vs 26 |
| 8 | offset map monotonic | OK |  |
| 9 | raw_slice reproduces raw text | OK | 'Thằng ' |
| 10 | exact match found | OK | term='ad ngu' |
| 11 | match_type is exact | OK | exact |
| 12 | raw_span == input text | OK | 'ad ngu' |
| 13 | scores in [0, 1] | OK | rule_score=0.7469 |
| 14 | no double count for a span+label | OK |  |
| 15 | base/plain/compact views built | OK | base,compact,plain |
| 16 | classify 'c0n ch0'->'con chó' | OK | leet, got ('obfuscated_combined', 'accentless+leet') |
| 17 | classify 'con cho'->'con chó' | OK | accentless, got ('accentless_alias', 'accentless') |
| 18 | classify 'nguuu'->'ngu' | OK | collapse, got ('repeat_collapsed', 'collapse') |
| 19 | classify 'n g u'->'ngu' | OK | separator, got ('separator_inserted', 'separators') |
| 20 | bounded_levenshtein distance | OK |  |
| 21 | handles empty string | OK | n_matches=0 |
| 22 | handles None | OK | n_matches=0 |
| 23 | handles very long text | OK | n_matches=200 truncated=True |
| 24 | eval validation (60 rows) | OK | macro_f1=0.3045 acc=0.5333 HATE_f1=0.2222 OFF_f1=0.0000 (0.30 ms/row) |

## End-to-end (argmax, no threshold tuning)

| Split | Rows | macro-F1 | accuracy | HATE-F1 | OFF-F1 | CLEAN-F1 | ms/row |
|-------|------|----------|----------|---------|--------|----------|--------|
| validation | 60 | 0.3045 | 0.5333 | 0.2222 | 0.0000 | 0.6914 | 0.2958 |
