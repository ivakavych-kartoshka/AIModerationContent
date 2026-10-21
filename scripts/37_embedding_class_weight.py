"""Step 37 - TRAIN embedding (fastText) tren 3 dataset de cham "trong so thien class".

    python scripts/37_embedding_class_weight.py
    python scripts/37_embedding_class_weight.py --epochs 10 --dim 100

Y tuong cua mentor (dien giai)
-----------------------------
    1. Tim tokenize cho tieng Viet            -> underthesea (script 34)
    2. Train voi du lieu lon gan voi minh     -> fastText unsupervised tren
                                                 ViHSD + ViCM + Bontori
    3. Sau khi tokenize, ghi lai trong so
       thien ve class nao                     -> vector term gan voi tam cua
                                                 tung class den muc nao
    4. Filter bot                             -> giu term co do nghieng manh
    5. Train xem no nam o label nao           -> nearest-class-centroid

Khac biet voi cach hien tai
---------------------------
`src/data/corpus_lexicon.py` chi DEM document frequency roi loc bang TF-IDF -
khong hoc gi tu ngu canh. Script nay HOC THAT: fastText hoc vector cho tung
token tu 135k dong van ban mang xa hoi tieng Viet, nen hai tu dong nghia/xuat
hien cung ngu canh se co vector gan nhau. Nho vay term moi (chua tung thay)
van duoc cham diem qua vector cua no.

Cach cham "trong so thien class"
--------------------------------
Voi moi class c (HATE/OFFENSIVE/CLEAN):
    centroid_c = trung binh vector cua tat ca token trong cac cau thuoc class c
    cos_c(term) = cosine(vector(term), centroid_c)
Trong so thien class = phan phoi mem (softmax) tren 3 cos, va:
    class_margin = cos(class_thang) - cos(class_nhi)

Filter: giu term co class_margin >= --min-margin va argmax != CLEAN.

Outputs (outputs/lexicon_experiments/fasttext_embedding/)
    corpus_tokenized.txt          corpus da tokenize (dau vao fastText)
    fasttext_model.bin/.vec       model da train
    term_class_scores.csv         trong so thien class cua tung term ung vien
    class_weight_report.md/json   tom tat
Khong ghi de lexicon chinh thuc (plan section 44).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
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
    _existing_lexicon,
    _load_rows_bontori,
    _load_rows_vicm,
    _load_rows_vihsd,
)

LOGGER = get_logger("sensitiveai.scripts.37_embedding_class_weight")

OUT_DIR = Path("outputs/lexicon_experiments/fasttext_embedding")
DEFAULT_CANDIDATE_DIR = Path("outputs/lexicon_experiments/vi_tokenizer")
CLASSES = ("HATE", "OFFENSIVE", "CLEAN")
MIN_TOKEN_LEN = 3  # ky tu (bo token qua ngan, thuong la rac)


# --------------------------------------------------------------------------- #
# 1. tokenize tieng Viet
# --------------------------------------------------------------------------- #
def tokenize_vi(text: str) -> List[str]:
    """underthesea word_tokenize; tu ghep gop bang '_' -> MOT token."""
    from underthesea import word_tokenize

    from src.data.lexicon import normalize_term

    normalized = normalize_term(text)
    if not normalized:
        return []
    try:
        segmented = word_tokenize(normalized, format="text")
    except Exception:
        segmented = normalized
    tokens: List[str] = []
    for raw in segmented.split():
        token = raw.strip()
        if len(token.replace("_", "")) < MIN_TOKEN_LEN:
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        tokens.append(token)
    return tokens


# --------------------------------------------------------------------------- #
# 2. corpus -> file tokenized + tap token theo class
# --------------------------------------------------------------------------- #
def build_corpus(rows: Sequence[Tuple[str, str]]) -> Tuple[List[str], Dict[str, Counter]]:
    """Tra ve (cac dong da tokenize, tap token xuat hien theo tung class)."""
    lines: List[str] = []
    tokens_by_class: Dict[str, Counter] = {c: Counter() for c in CLASSES}
    for text, lab in rows:
        key = lab.upper()
        if key not in CLASSES:
            continue
        tokens = tokenize_vi(text)
        if not tokens:
            continue
        lines.append(" ".join(tokens))
        tokens_by_class[key].update(set(tokens))  # document frequency
    return lines, tokens_by_class


# --------------------------------------------------------------------------- #
# 3. train fastText
# --------------------------------------------------------------------------- #
def train_fasttext(lines: Sequence[str], out_dir: Path, dim: int, epochs: int,
                   min_count: int, window: int) -> Any:
    import fasttext

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus_tokenized.txt"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    model_path = out_dir / "fasttext_model"
    LOGGER.info("[37] training fastText: %d lines, dim=%d epochs=%d", len(lines), dim, epochs)
    model = fasttext.train_unsupervised(
        str(corpus_path),
        model="skipgram",
        dim=dim,
        epoch=epochs,
        minCount=min_count,
        ws=window,
        minn=3,
        maxn=5,
        thread=4,
        verbose=0,
    )
    model.save_model(str(model_path) + ".bin")
    return model


# --------------------------------------------------------------------------- #
# 4. centroid theo class + trong so thien class
# --------------------------------------------------------------------------- #
def class_centroids(model: Any, tokens_by_class: Dict[str, Counter]) -> Dict[str, np.ndarray]:
    """Trung binh vector cua token trong tung class, co trong so document frequency."""
    centroids: Dict[str, np.ndarray] = {}
    for cls in CLASSES:
        counter = tokens_by_class[cls]
        if not counter:
            continue
        vecs = []
        weights = []
        for token, dfv in counter.items():
            vec = model.get_word_vector(token)
            if not np.any(vec):
                continue
            vecs.append(vec)
            weights.append(float(dfv))
        if not vecs:
            continue
        arr = np.vstack(vecs)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        centroids[cls] = (arr * w[:, None]).sum(axis=0)
        LOGGER.info("[37] centroid %s tu %d token", cls, len(vecs))
    return centroids


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_terms(model: Any, centroids: Dict[str, np.ndarray],
                terms: Sequence[str]) -> List[Dict[str, Any]]:
    """Cham cho tung term: cos voi moi class, phan phoi mem, class_margin."""
    order = [c for c in CLASSES if c in centroids]
    out: List[Dict[str, Any]] = []
    for term in terms:
        vec = model.get_word_vector(term)
        cos = {c: _cosine(vec, centroids[c]) for c in order}
        vals = np.asarray([cos[c] for c in order], dtype=float)
        # softmax co nhiet do de bien cos (rat gan nhau) thanh trong so
        scaled = (vals - vals.max()) / 0.05
        exp = np.exp(scaled)
        soft = exp / exp.sum()
        soft_map = {c: float(p) for c, p in zip(order, soft)}
        ranked = sorted(order, key=lambda c: cos[c], reverse=True)
        best, second = ranked[0], ranked[1] if len(ranked) > 1 else ranked[0]
        out.append({
            "term": term,
            **{f"cos_{c.lower()}": round(cos[c], 6) for c in order},
            **{f"weight_{c.lower()}": round(soft_map[c], 6) for c in order},
            "pred_label": best,
            "runner_up": second,
            "class_margin": round(cos[best] - cos[second], 6),
        })
    return out


# --------------------------------------------------------------------------- #
# 5. main
# --------------------------------------------------------------------------- #
def load_rows(splits: Sequence[str]) -> List[Tuple[str, str]]:
    legacy = tuple(splits) == DEFAULT_SPLITS_VIHSD
    vihsd = _load_rows_vihsd(PROCESSED_DIR / "vihsd", splits)
    vicm_files = DEFAULT_FILES_VICM if legacy else tuple(f"{s}.xlsx" for s in splits)
    bontori_files = DEFAULT_FILES_BONTORI if legacy else tuple(f"tdtu_{s}.parquet" for s in splits)
    vicm = _load_rows_vicm(RAW_DIR / "additional" / "vicm-code-mixed", vicm_files)
    bontori = _load_rows_bontori(RAW_DIR / "additional" / "bontori-vietnamese-hsd-cleaned", bontori_files)
    return vihsd + vicm + bontori


def read_terms(paths: Iterable[Path]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            term = line.strip()
            if term and term not in seen:
                seen.add(term)
                ordered.append(term)
    return ordered


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train fastText + cham trong so thien class.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Giu term co class_margin >= nguong nay.")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    rows = load_rows(splits)
    LOGGER.info("[37] tong so dong: %d", len(rows))

    lines, tokens_by_class = build_corpus(rows)
    LOGGER.info("[37] corpus tokenized: %d dong", len(lines))
    for cls in CLASSES:
        LOGGER.info("[37]   %s: %d token duy nhat", cls, len(tokens_by_class[cls]))

    model = train_fasttext(lines, out_dir, args.dim, args.epochs, args.min_count, args.window)
    centroids = class_centroids(model, tokens_by_class)

    # term ung vien: 668 hien tai + 820 term che do word cua script 34
    existing_terms = read_terms([
        LEXICON_PROCESSED_DIR / "verified_lexicon.csv",
    ])
    # doc term tu CSV that (verified_lexicon.csv co header)
    verified = pd.read_csv(LEXICON_PROCESSED_DIR / "verified_lexicon.csv")
    existing_terms = verified["term"].astype(str).tolist()
    word_terms = read_terms([
        DEFAULT_CANDIDATE_DIR / "word_mode_vihsd.txt",
        DEFAULT_CANDIDATE_DIR / "word_mode_vicm.txt",
        DEFAULT_CANDIDATE_DIR / "word_mode_bontori.txt",
    ])
    candidates = list(dict.fromkeys(existing_terms + word_terms))
    LOGGER.info("[37] cham diem %d term ung vien", len(candidates))

    scored = score_terms(model, centroids, candidates)
    df = pd.DataFrame(scored)
    df.to_csv(out_dir / "term_class_scores.csv", index=False, encoding="utf-8")

    # ---- filter -------------------------------------------------------------
    kept = df[(df["pred_label"] != "CLEAN") & (df["class_margin"] >= args.min_margin)].copy()
    kept = kept.sort_values(["pred_label", "class_margin"], ascending=[True, False])
    kept.to_csv(out_dir / "term_class_scores_kept.csv", index=False, encoding="utf-8")
    LOGGER.info("[37] giu %d / %d term (pred != CLEAN, margin >= %.3f)",
                len(kept), len(df), args.min_margin)

    # ---- report -------------------------------------------------------------
    dist = df["pred_label"].value_counts().to_dict()
    kept_dist = kept["pred_label"].value_counts().to_dict()
    summary = {
        "corpus_lines": len(lines),
        "unique_tokens": {c: len(tokens_by_class[c]) for c in CLASSES},
        "candidates": len(df),
        "pred_distribution": dist,
        "kept": len(kept),
        "kept_distribution": kept_dist,
        "params": {"dim": args.dim, "epochs": args.epochs, "min_count": args.min_count,
                   "window": args.window, "min_margin": args.min_margin, "splits": list(splits)},
    }
    (out_dir / "class_weight_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines_md: List[str] = []
    add = lines_md.append
    add("# Train embedding (fastText) + trong so thien class\n")
    add(f"> Corpus: ViHSD + ViCM + Bontori ({len(lines)} dong da tokenize bang underthesea). "
        f"fastText skipgram dim={args.dim}, epochs={args.epochs}. "
        "Centroid moi class = trung binh vector token trong class do (trong so DF).\n")
    add("## Phan bo term ung vien theo label du doan\n")
    add("| Label | Tat ca ung vien | Sau filter |")
    add("|---|---|---|")
    for cls in CLASSES:
        add(f"| {cls} | {dist.get(cls, 0)} | {kept_dist.get(cls, 0)} |")
    add(f"| **Tong** | **{len(df)}** | **{len(kept)}** |")
    add("\n## Vi du term nghieng ve HATE (margin cao nhat)\n")
    for cls in ("HATE", "OFFENSIVE"):
        sub = df[df["pred_label"] == cls].nlargest(15, "class_margin")
        add(f"### {cls} ({dist.get(cls, 0)} term)\n")
        add("| term | cos_hate | cos_off | cos_clean | margin |")
        add("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            add(f"| `{r['term']}` | {r['cos_hate']:.4f} | {r['cos_offensive']:.4f} | "
                f"{r['cos_clean']:.4f} | {r['class_margin']:+.4f} |")
        add("")
    add("## Cach doc\n")
    add("- `weight_*` = phan phoi mem: term nay 'thien' ve class nao theo ngu canh hoc duoc.")
    add("- `class_margin` = do ro rang; margin nho = term trung tinh, khong nen giu.")
    add("- Day moi la buoc **cham diem**; buoc do F1 tren validation se lam o script 38.")
    add("- Chua dung vao lexicon chinh thuc (plan section 44).")
    (out_dir / "class_weight_report.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    LOGGER.info("[37] wrote %s/class_weight_report.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
