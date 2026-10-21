"""Step 34 - so sanh tokenize tieng Viet (underthesea) khi TACH LEXICON tu 3 dataset.

    python scripts/34_lexicon_vi_tokenizer.py
    python scripts/34_lexicon_vi_tokenizer.py --splits train --min-toxic-df 5 --max-clean-df 1

Boi canh
--------
`src/data/corpus_lexicon.py` hien tai tach term bang `normalized.split()` tren
khoang trang, tuc la cat theo AM TIET (syllable): "con cho" -> ["con", "cho"]
roi ghep lai thanh bigram "con cho". Cach nay vo tinh dung cho tu 2 am tiet
lien ke nhung KHONG bao gio tao duoc trigram ("do ngu ngoc") va khong biet
dau la tu ghep that.

Script nay chay SONG SONG mot nhanh thi nghiem: cung du lieu, cung nguong
DF/TF-IDF, nhung term duoc tach bang `underthesea.word_tokenize` (tu ghep ->
mot token "khon_nan"). Ket qua ghi vao thu muc RIENG, khong ghi de lexicon
dang dung (plan section 44: khong pha vo so lieu da dong bang).

Hai che do tach duoc so sanh:

  syllable : cach hien tai (split tren khoang trang) - baseline
  word     : underthesea word_tokenize (tu ghep gop bang "_")

Voi moi che do, script in ra: so term tim duoc, so term trung nhau, vi du term
moi chi xuat hien o mot che do - de thay tokenizer tieng Viet co tim them duoc
term gi ma che do cu bo sot.

Outputs (outputs/lexicon_experiments/vi_tokenizer/)
    word_mode_<source>.txt        term tach bang underthesea
    syllable_mode_<source>.txt    term tach bang split (chay lai de doi chieu)
    comparison.md                 bang so sanh + vi du
    comparison.json               cung so lieu dang may doc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.common.constants import LABEL2ID, LEXICON_PROCESSED_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.data.corpus_lexicon import (  # noqa: E402
    DEFAULT_FILES_BONTORI,
    DEFAULT_FILES_VICM,
    DEFAULT_SPLITS_VIHSD,
    _existing_lexicon,
    _load_rows_bontori,
    _load_rows_vicm,
    _load_rows_vihsd,
    strip_token,
)
from src.data.lexicon import normalize_term  # noqa: E402
from src.data.tfidf_filter import TfidfConfig, combined_tfidf, evaluate as tfidf_evaluate  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.34_lexicon_vi_tokenizer")

DEFAULT_OUT = Path("outputs/lexicon_experiments/vi_tokenizer")


def tokenize_syllable(text: str) -> List[str]:
    """Che do hien tai: cat tren khoang trang (giu nguyen de doi chieu)."""
    normalized = normalize_term(text)
    if not normalized:
        return []
    tokens: List[str] = []
    for raw in normalized.split():
        token = strip_token(raw)
        if not token or not any(ch.isalpha() for ch in token):
            continue
        tokens.append(token)
    return tokens


def tokenize_word(text: str) -> List[str]:
    """Che do tieng Viet: underthesea word_tokenize, tu ghep gop bang '_'.

    Tra ve token da normalize (bo dau khong - giu dau nhu normalize_term) va
    gop tu ghep thanh MOT token de bigram/trigram di theo don vi TU, khong phai
    am tiet.
    """
    from underthesea import word_tokenize

    normalized = normalize_term(text)
    if not normalized:
        return []
    try:
        segmented = word_tokenize(normalized, format="text")
    except Exception:  # pragma: no cover - tokenizer hiem khi loi tren text la
        return tokenize_syllable(normalized)
    tokens: List[str] = []
    for raw in segmented.split():
        # "khon_nan" -> giu nguyen duoi dang mot token; strip dau cau o 2 bien
        token = strip_token(raw)
        if not token or not any(ch.isalpha() for ch in token):
            continue
        # normalize lai tung token de dong nhat voi lexicon (NFC, lowercase)
        token = normalize_term(token)
        if token:
            tokens.append(token)
    return tokens


def ngrams(tokens: Sequence[str], min_len: int = 3) -> Set[str]:
    """Unigram + bigram + trigram (trigram la cai che do cu KHONG co)."""
    out: Set[str] = set(tokens)
    for size in (2, 3):
        for i in range(len(tokens) - size + 1):
            piece = " ".join(tokens[i : i + size])
            if len(piece.replace(" ", "")) >= min_len:
                out.add(piece)
    return out


def extract_for_source(
    rows: Iterable[Tuple[str, str]],
    existing: Set[str],
    tokenizer,
    min_toxic_df: int,
    max_clean_df: int,
    tfidf_cfg: TfidfConfig,
) -> Dict[str, Any]:
    """Chay lai dung cong thuc DF + TF-IDF cua corpus_lexicon tren mot tokenizer."""
    toxic_df: Counter = Counter()
    clean_df: Counter = Counter()
    rows_by_label: Counter = Counter()
    for text, lab in rows:
        key = lab.upper()
        if key not in ("HATE", "OFFENSIVE", "CLEAN"):
            continue
        rows_by_label[key] += 1
        tokens = tokenizer(text)
        if not tokens:
            continue
        terms = ngrams(tokens)
        if key in ("HATE", "OFFENSIVE"):
            toxic_df.update(terms)
        else:
            clean_df.update(terms)
    n_toxic = rows_by_label.get("HATE", 0) + rows_by_label.get("OFFENSIVE", 0)
    n_clean = rows_by_label.get("CLEAN", 0)
    kept: Set[str] = set()
    rejected: Counter = Counter()
    for term, dfv in toxic_df.items():
        if dfv < min_toxic_df or clean_df.get(term, 0) > max_clean_df:
            continue
        if term in existing or len(term.replace(" ", "")) < 2:
            continue
        stats = combined_tfidf(dfv, clean_df.get(term, 0), n_toxic, n_clean, tfidf_cfg)
        ok, reason = tfidf_evaluate(stats.tfidf_toxic, stats.ratio, stats.score, tfidf_cfg)
        if not ok:
            rejected[reason] += 1
            continue
        kept.add(term)
    return {"terms": kept, "rows_by_label": dict(rows_by_label), "rejected": dict(rejected)}


def _write(path: Path, terms: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for term in sorted(terms):
            fh.write(term + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="So sanh tokenize tieng Viet khi tach lexicon.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--min-toxic-df", type=int, default=5)
    parser.add_argument("--max-clean-df", type=int, default=1)
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--show-terms", type=int, default=25, help="So vi du term in ra moi nguon.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    tfidf_cfg = TfidfConfig()
    existing = _existing_lexicon(LEXICON_PROCESSED_DIR)
    LOGGER.info("[34] existing lexicon terms: %d", len(existing))

    # ---- load the three corpora once ---------------------------------------
    legacy = splits == DEFAULT_SPLITS_VIHSD
    vihsd_rows = _load_rows_vihsd(PROCESSED_DIR / "vihsd", splits)
    vicm_files: Sequence[str] = DEFAULT_FILES_VICM if legacy else tuple(f"{s}.xlsx" for s in splits)
    bontori_files: Sequence[str] = DEFAULT_FILES_BONTORI if legacy else tuple(f"tdtu_{s}.parquet" for s in splits)
    vicm_rows = _load_rows_vicm(RAW_DIR / "additional" / "vicm-code-mixed", vicm_files)
    bontori_rows = _load_rows_bontori(RAW_DIR / "additional" / "bontori-vietnamese-hsd-cleaned", bontori_files)

    sources = {"vihsd": vihsd_rows, "vicm": vicm_rows, "bontori": bontori_rows}
    modes = {"syllable": tokenize_syllable, "word": tokenize_word}

    report: Dict[str, Any] = {"min_toxic_df": args.min_toxic_df, "max_clean_df": args.max_clean_df,
                              "splits": list(splits), "sources": {}}
    all_terms: Dict[str, Dict[str, Set[str]]] = {}
    for source, rows in sources.items():
        LOGGER.info("[34] source %s: %d rows", source, len(rows))
        all_terms[source] = {}
        for mode, tokenizer in modes.items():
            result = extract_for_source(rows, existing, tokenizer, args.min_toxic_df,
                                        args.max_clean_df, tfidf_cfg)
            all_terms[source][mode] = result["terms"]
            _write(out_dir / f"{mode}_mode_{source}.txt", result["terms"])
            LOGGER.info("[34]   %-8s -> %d terms", mode, len(result["terms"]))
        report["sources"][source] = {
            mode: len(all_terms[source][mode]) for mode in modes
        }

    # ---- compare ------------------------------------------------------------
    totals = {mode: 0 for mode in modes}
    for source in sources:
        for mode in modes:
            totals[mode] += len(all_terms[source][mode])
    report["totals"] = totals

    lines: List[str] = []
    add = lines.append
    add("# Tách lexicon từ ViHSD / ViCM / bontori — so sánh tokenizer tiếng Việt\n")
    add(f"> Ngưỡng: min_toxic_df={args.min_toxic_df}, max_clean_df={args.max_clean_df}, "
        f"TF-IDF gate mặc định; splits={','.join(splits)}. "
        "Che do `word` = underthesea word_tokenize (từ ghép → 1 token, có trigram). "
        "Che do `syllable` = cách hiện tại (split khoảng trắng, chỉ unigram+bigram).\n")
    add("## 1. So luong term moi tim duoc\n")
    add("| Nguồn | `syllable` (hiện tại) | `word` (underthesea) | Chỉ có ở `word` | Chỉ có ở `syllable` | Trùng nhau |")
    add("|---|---|---|---|---|---|")
    for source in sources:
        syl = all_terms[source]["syllable"]
        wrd = all_terms[source]["word"]
        add(f"| {source} | {len(syl)} | {len(wrd)} | {len(wrd - syl)} | {len(syl - wrd)} | {len(syl & wrd)} |")
    add(f"| **Tổng** | **{totals['syllable']}** | **{totals['word']}** | "
        f"**{sum(len(all_terms[s]['word'] - all_terms[s]['syllable']) for s in sources)}** | "
        f"**{sum(len(all_terms[s]['syllable'] - all_terms[s]['word']) for s in sources)}** | "
        f"**{sum(len(all_terms[s]['syllable'] & all_terms[s]['word']) for s in sources)}** |")
    add("\n## 2. Ví dụ term mà CHỈ chế độ `word` (tokenizer tiếng Việt) tìm được\n")
    for source in sources:
        only_word = sorted(all_terms[source]["word"] - all_terms[source]["syllable"])
        add(f"### {source} — {len(only_word)} term\n")
        if only_word:
            add("```")
            add("   ".join(only_word[: args.show_terms]))
            add("```")
        else:
            add("_(không có)_")
        add("")
        report["sources"][source]["only_word_examples"] = only_word[: args.show_terms]
    add("## 3. Ví dụ term mà CHỈ chế độ `syllable` (hiện tại) tìm được\n")
    for source in sources:
        only_syl = sorted(all_terms[source]["syllable"] - all_terms[source]["word"])
        add(f"### {source} — {len(only_syl)} term\n")
        if only_syl:
            add("```")
            add("   ".join(only_syl[: args.show_terms]))
            add("```")
        else:
            add("_(không có)_")
        add("")
    add("## 4. Cách đọc\n")
    add("- `word` nhiều term hơn hẳn = tokenizer tiếng Việt tách từ ghép đúng → tìm được term dài (trigram) mà `split` bỏ sót.")
    add("- Term ở cột 'chỉ có ở word' thường là cụm 2-3 từ ghép thật (vd `khốn_nạn` + ngữ cảnh).")
    add("- Term ở cột 'chỉ có ở syllable' thường là mảnh âm tiết rời không mang nghĩa.")
    add("- **Chưa** đưa các term này vào lexicon chính thức — đây là nhánh thí nghiệm để so sánh.")

    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[34] totals: syllable=%d word=%d", totals["syllable"], totals["word"])
    LOGGER.info("[34] wrote %s/comparison.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
