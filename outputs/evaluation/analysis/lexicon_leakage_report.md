# Lexicon-level leakage audit (held-out test sets)

## VI lexicon

- lexicon file: data/processed/lexicon/verified_lexicon.csv
- lexicon terms: 668
- rows with bad source split (test/validation/dev): 0
- source split counts: {'train': 668}
- source dataset counts: {'bontori;vicm;vihsd': 414, 'bontori;vihsd': 95, 'bontori': 74, 'bontori;vicm': 65, 'vicm': 20}
- labels: {'HATE': 594, 'OFFENSIVE': 74}
- exact full-comment matches with held-out test: 13
- terms firing in test but not train: 0

## EN lexicon

- lexicon file: data/processed/lexicon/verified_lexicon_en_combined.csv
- lexicon terms: 818
- rows with bad source split (test/validation/dev): 0
- source split counts: {'train': 778, 'manual': 40}
- source dataset counts: {'davidson': 778, 'seed': 40}
- labels: {'OFFENSIVE': 356, 'HATE': 86, 'CLEAN': 376}
- exact full-comment matches with held-out test: 0
- terms firing in test but not train: 0

