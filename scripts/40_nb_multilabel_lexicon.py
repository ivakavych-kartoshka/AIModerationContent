"""Step 40 - MO HINH XAC SUAT (Naive Bayes) + MULTI-LABEL cho term lexicon.

    python scripts/40_nb_multilabel_lexicon.py
    python scripts/40_nb_multilabel_lexicon.py --min-prob 0.25 --smoothing 1.0

Y tuong cua mentor (dien giai)
-----------------------------
    "co the nam o nhieu label + dung mo hinh xac suat de tinh xem tu do thuoc
     nhan nao?"

Hai huong truoc deu that bai vi hai ly do thiet ke:
    - script 35/36 (tokenizer): van la DEM document frequency + TF-IDF
    - script 37/38 (fastText):  dung cosine roi ep ve MOT nhan bang argmax

Script nay sua ca hai:
    1. MO HINH XAC SUAT THAT: Naive Bayes tren bag-of-words.
       P(term | class c) = (df_c(term) + alpha) / (N_c + alpha * V)
       P(term)           = (df_all(term) + alpha) / (N_all + alpha * V)
       log-odds(c)       = log P(term|c) - log P(term)     (diem "thien class")
       -> chuan hoa qua softmax thanh phan phoi P(c | term)
    2. MULTI-LABEL THAT: giu MOI class co P(c|term) >= --min-prob, khong argmax.
       Matcher trong src/rule_engine/matcher.py da ho tro 1 term nhieu nhan
       (`for label in entry.labels`), nen day la tan dung ha tang CO SAN.

Vi sao NB chu khong cosine?
    Cosine do "hai vector gan nhau khong" -> moi class deu ~0.44-0.49, khong
    tach duoc. NB do "term nay xuat hien trong class nay NHIEU HON HAN cac
    class khac khong" (log-odds) -> tach duoc term co tin hieu that.

Ngung bay thong ke can tranh
    Vi class rat lech (CLEAN 83k vs HATE 9.6k), P(c) tien nghiem se keo moi
    term ve CLEAN. Vi vay ta dung LO-ODDS (da chia cho P(term) chung) chu
    khong dung posterior tho; lo-odds la "lift" nen khong bi lech class.

Outputs (outputs/lexicon_experiments/nb_multilabel/)
    term_probabilities.csv      P(c|term) cho tung term ung vien
    lexicon_nb_multilabel.csv   lexicon sinh ra (co term mang 2 nhan)
    nb_report.md/json           tom tat
Khong ghi de lexicon chinh thuc (plan section 44).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.constants import LEXICON_PROCESSED_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.data.corpus_lexicon import (  # noqa: E402
    DEFAULT_FILES_BONTORI,
    DEFAULT_FILES_VICM,
    DEFAULT_SPLITS_VIHSD,
    _load_rows_bontori,
    _load_rows_vicm,
    _load_rows_vihsd,
)

LOGGER = get_logger("sensitiveai.scripts.40_nb_multilabel_lexicon")

OUT_DIR = Path("outputs/lexicon_experiments/nb_multilabel")
CANDIDATE_DIRS = (
    Path("outputs/lexicon_experiments/vi_tokenizer"),
    Path("outputs/lexicon_experiments/fasttext_embedding"),
)
CLASSES = ("HATE", "OFFENSIVE", "CLEAN")
SEVERITY = {"HATE": 0.9, "OFFENSIVE": 0.6}


def tokenize_row(text: str) -> List[str]:
    """Token hoa don gian (syllable + bigram/trigram) - KHONG dung underthesea.

    Ly do: script 36 da chung minh tokenizer chuan khong giup rule engine, va
    NB can term xuat hien ca trong text nhieu (matcher dung word boundary).
    """
    from src.data.lexicon import normalize_term

    normalized = normalize_term(text)
    if not normalized:
        return []
    toks = [t for t in normalized.split() if any(ch.isalpha() for ch in t)]
    return toks


def count_df(rows: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, Counter], Counter, Counter]:
    """Dem document frequency cho tung class + tong + so dong moi class."""
    df_by_class: Dict[str, Counter] = {c: Counter() for c in CLASSES}
    df_all: Counter = Counter()
    n_rows: Counter = Counter()
    for text, lab in rows:
        key = lab.upper()
        if key not in CLASSES:
            continue
        n_rows[key] += 1
        toks = tokenize_row(text)
        if not toks:
            continue
        terms: Set[str] = set(toks)
        for size in (2, 3):
            for i in range(len(toks) - size + 1):
                terms.add(" ".join(toks[i:i + size]))
        df_by_class[key].update(terms)
        df_all.update(terms)
    return df_by_class, df_all, n_rows


def nb_probabilities(df_by_class: Dict[str, Counter], df_all: Counter, n_rows: Counter,
                     terms: Sequence[str], alpha: float) -> List[Dict[str, Any]]:
    """P(c|term) qua lo-odds Naive Bayes; tra ve tat ca class (khong argmax)."""
    n_all = sum(n_rows.values())
    vocab = len(df_all)
    prior = {c: n_rows[c] / n_all for c in CLASSES}
    out: List[Dict[str, Any]] = []
    for term in terms:
        d_all = df_all.get(term, 0)
        if d_all == 0:
            continue
        # p(term) chung - dung lam mau so cua lo-odds
        p_term = (d_all + alpha) / (n_all + alpha * vocab)
        log_odds: Dict[str, float] = {}
        for c in CLASSES:
            d_c = df_by_class[c].get(term, 0)
            p_c = (d_c + alpha) / (n_rows[c] + alpha * vocab)
            log_odds[c] = math.log(p_c) - math.log(p_term)
        # posterior khong-lech: chi dung phan vuot troi (log-odds da tru P(term))
        vals = np.asarray([log_odds[c] for c in CLASSES], dtype=float)
        exp = np.exp(vals - vals.max())
        prob = exp / exp.sum()
        prob_map = {c: float(p) for c, p in zip(CLASSES, prob)}
        ranked = sorted(CLASSES, key=lambda c: prob_map[c], reverse=True)
        out.append({
            "term": term,
            "df_all": d_all,
            **{f"df_{c.lower()}": df_by_class[c].get(term, 0) for c in CLASSES},
            **{f"logodds_{c.lower()}": round(log_odds[c], 4) for c in CLASSES},
            **{f"prob_{c.lower()}": round(prob_map[c], 6) for c in CLASSES},
            "top_label": ranked[0],
            "second_label": ranked[1],
            "top_prob": round(prob_map[ranked[0]], 6),
        })
    return out


def select_multilabel(df: pd.DataFrame, min_prob: float) -> pd.DataFrame:
    """Giu MOI class co prob >= min_prob (multi-label). Bo term khong co nhan toxic."""
    rows = []
    for i, r in df.iterrows():
        labels = [c for c in ("HATE", "OFFENSIVE") if r[f"prob_{c.lower()}"] >= min_prob]
        if not labels:
            continue
        conf = float(max(r["prob_hate"], r["prob_offensive"]))
        rows.append({
            "entry_id": f"nb-{i:05d}",
            "term": r["term"],
            "canonical_form": r["term"],
            "lexical_labels": ";".join(labels),
            "severity": max(SEVERITY[l] for l in labels),
            "confidence": round(conf, 4),
            "context_required": "false",
            "source_datasets": "vihsd;vicm;bontori",
            "source_split": "train",
            "review_status": "auto",
            "dictionary_version": "nb-multilabel-2026-10-20",
        })
    return pd.DataFrame(rows)


def read_terms(paths: Iterable[Path]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def load_rows(splits: Sequence[str]) -> List[Tuple[str, str]]:
    legacy = tuple(splits) == DEFAULT_SPLITS_VIHSD
    vihsd = _load_rows_vihsd(PROCESSED_DIR / "vihsd", splits)
    vicm_files = DEFAULT_FILES_VICM if legacy else tuple(f"{s}.xlsx" for s in splits)
    bontori_files = DEFAULT_FILES_BONTORI if legacy else tuple(f"tdtu_{s}.parquet" for s in splits)
    vicm = _load_rows_vicm(RAW_DIR / "additional" / "vicm-code-mixed", vicm_files)
    bontori = _load_rows_bontori(RAW_DIR / "additional" / "bontori-vietnamese-hsd-cleaned", bontori_files)
    return vihsd + vicm + bontori


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NB xac suat + multi-label cho lexicon.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--smoothing", type=float, default=1.0, help="alpha Laplace.")
    parser.add_argument("--min-prob", type=float, default=0.25,
                        help="Nguong P(c|term) de giu 1 nhan (cao hon -> it nhan hon).")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    rows = load_rows(splits)
    LOGGER.info("[40] tong dong: %d", len(rows))

    df_by_class, df_all, n_rows = count_df(rows)
    LOGGER.info("[40] df: HATE=%d OFF=%d CLEAN=%d, vocab=%d",
                len(df_by_class["HATE"]), len(df_by_class["OFFENSIVE"]),
                len(df_by_class["CLEAN"]), len(df_all))

    verified = pd.read_csv(LEXICON_PROCESSED_DIR / "verified_lexicon.csv")
    existing = verified["term"].astype(str).tolist()
    word_terms = read_terms([
        CANDIDATE_DIRS[0] / "word_mode_vihsd.txt",
        CANDIDATE_DIRS[0] / "word_mode_vicm.txt",
        CANDIDATE_DIRS[0] / "word_mode_bontori.txt",
    ])
    candidates = list(dict.fromkeys(existing + word_terms))
    LOGGER.info("[40] %d term ung vien", len(candidates))

    scored = nb_probabilities(df_by_class, df_all, n_rows, candidates, args.smoothing)
    sdf = pd.DataFrame(scored)
    sdf.to_csv(out_dir / "term_probabilities.csv", index=False, encoding="utf-8")
    LOGGER.info("[40] cham diem %d term (co trong corpus)", len(sdf))

    ml = select_multilabel(sdf, args.min_prob)
    ml.to_csv(out_dir / "lexicon_nb_multilabel.csv", index=False, encoding="utf-8")
    n_multi = int((ml["lexical_labels"].str.contains(";")).sum())
    LOGGER.info("[40] multi-label lexicon: %d entry (trong do %d term mang 2 nhan)",
                len(ml), n_multi)

    top1 = sdf["top_label"].value_counts().to_dict()
    summary = {
        "corpus_rows": len(rows), "vocab": len(df_all), "smoothing": args.smoothing,
        "min_prob": args.min_prob, "candidates_scored": len(sdf),
        "top_label_distribution": top1, "kept_entries": len(ml),
        "multi_label_entries": n_multi, "splits": list(splits),
    }
    (out_dir / "nb_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    lines: List[str] = []
    add = lines.append
    add("# Mo hinh xac suat (Naive Bayes) + multi-label cho lexicon\n")
    add(f"> Corpus: ViHSD+ViCM+Bontori ({len(rows)} dong). Laplace alpha={args.smoothing}, "
        f"nguong multi-label P>={args.min_prob}. Khong cham test.\n")
    add("## Phan bo top-1 label (truoc khi multi-label hoa)\n")
    add("| Top-1 label | So term |")
    add("|---|---|")
    for cls in CLASSES:
        add(f"| {cls} | {top1.get(cls, 0)} |")
    add("\n## Sau multi-label hoa\n")
    add(f"- Tong entry giu lai: **{len(ml)}** (prob HATE hoac OFF >= {args.min_prob})")
    add(f"- Trong do **{n_multi}** entry mang CA HAI nhan `HATE;OFFENSIVE` "
        "(matcher se cong diem cho ca 2 class)")
    add("\n## Vi du term xac suat cao\n")
    add("| term | P(HATE) | P(OFF) | P(CLEAN) | nhan giu |")
    add("|---|---|---|---|---|")
    for label in ("HATE", "OFFENSIVE"):
        sub = sdf[sdf["top_label"] == label].nlargest(8, "top_prob")
        for _, r in sub.iterrows():
            kept = [c for c in ("HATE", "OFFENSIVE") if r[f"prob_{c.lower()}"] >= args.min_prob]
            add(f"| `{r['term']}` | {r['prob_hate']:.3f} | {r['prob_offensive']:.3f} | "
                f"{r['prob_clean']:.3f} | {';'.join(kept)} |")
    add("\n## Cach doc\n")
    add("- `prob_*` la posterior SOFTMAX cua lo-odds Naive Bayes (da bu tru class lech).")
    add("- Term mang 2 nhan = term xuat hien nhieu o CA HAI class (da nghia that, "
        "vd `dm`, `vl`) - day chinh la thu argmax truoc day lam mat.")
    add("- Buoc do F1 tren validation: script 41.")
    (out_dir / "nb_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("[40] wrote %s/nb_report.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
