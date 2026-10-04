"""Text normalisation for ViHSD (research plan, section 3).

Design rules taken directly from the plan:

* drop rows that have no text at all;
* handle ``null`` / ``NaN`` explicitly;
* ``lower()``;
* normalise whitespace, newlines and tabs;
* **keep Vietnamese diacritics** (never strip accents, never NFD-fold);
* never remove information that the model may need.

Everything that *deletes* content (URLs, mentions, repeated punctuation, ...)
is therefore **opt-in** and disabled by default. The defaults only perform
*lossless decoding repairs* (HTML entity unescape, invisible-character removal,
NFKC-ish whitespace folding) which would otherwise feed the tokenizer mojibake.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import pandas as pd

# --------------------------------------------------------------------------- #
# Compiled patterns
# --------------------------------------------------------------------------- #
#: Zero-width / bidi / soft-hyphen style characters that carry no linguistic signal.
INVISIBLE_RE = re.compile(
    "["
    "\u00ad"          # soft hyphen
    "\u200b\u200c\u200d"  # zero width space / non-joiner / joiner
    "\u2060"          # word joiner
    "\ufeff"          # byte order mark
    "\u200e\u200f"    # LTR / RTL marks
    "\u202a-\u202e"    # bidi embedding controls
    "\u2066-\u2069"    # bidi isolates
    "]"
)

#: All Unicode whitespace (incl. NBSP, thin space, ideographic space) -> " ".
ALL_SPACE_RE = re.compile(r"[\s\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")

#: Whitespace hugging punctuation that tokenizers treat inconsistently.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"[@#][\w.\-]+")
HTML_TAG_RE = re.compile(r"<[^>]{1,40}>")
REPEAT_PUNCT_RE = re.compile(r"([!?.])\1{2,}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")


@dataclass
class CleaningConfig:
    """Every knob of the cleaning pipeline, recorded in the run metadata."""

    lowercase: bool = True
    normalize_unicode: str = "NFC"          # NFC keeps Vietnamese diacritics composed
    unescape_html_entities: bool = True
    remove_invisible_chars: bool = True
    strip_tags: bool = True
    normalize_unicode_spaces: bool = True
    fix_punct_spacing: bool = True
    collapse_to_single_space: bool = True
    strip_edges: bool = True
    # --- opt-in, information-removing steps (OFF by default) ---
    remove_urls: bool = False
    remove_mentions: bool = False
    remove_repeated_punctuation: bool = False
    max_repeat_char: int = 0                # 0 = disabled
    min_chars: int = 1                      # rows shorter than this after cleaning are dropped

    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _visible_length(text: str) -> int:
    """Length in characters after invisible characters are removed."""
    return len(INVISIBLE_RE.sub("", text))


def clean_text(text: object, cfg: CleaningConfig | None = None) -> str | None:
    """Normalise a single text.

    Returns ``None`` when the input cannot be turned into usable text; callers
    use that to drop the row instead of feeding an empty string to the model.
    """
    cfg = cfg or CleaningConfig()

    # 1. null / NaN handling -------------------------------------------------
    if text is None:
        return None
    if isinstance(text, float) and pd.isna(text):
        return None
    if not isinstance(text, str):
        if isinstance(text, (bytes, bytearray)):
            try:
                text = text.decode("utf-8", errors="replace")
            except Exception:
                return None
        else:
            try:
                if pd.isna(text):
                    return None
            except (TypeError, ValueError):
                pass
            text = str(text)

    # 2. decoding repairs (lossless) ---------------------------------------
    if cfg.normalize_unicode:
        text = unicodedata.normalize(cfg.normalize_unicode, text)
    if cfg.unescape_html_entities:
        if "&" in text:
            text = html.unescape(text)
    if cfg.strip_tags:
        text = HTML_TAG_RE.sub(" ", text)
    if cfg.remove_invisible_chars:
        text = INVISIBLE_RE.sub("", text)

    # 3. case ---------------------------------------------------------------
    # NOTE: str.lower() is Unicode-aware and leaves Vietnamese diacritics
    # (e.g. "Ớ" -> "ớ") intact, unlike ASCII-only lowercasing.
    if cfg.lowercase:
        text = text.lower()

    # 4. opt-in deletions ---------------------------------------------------
    if cfg.remove_urls:
        text = URL_RE.sub(" ", text)
    if cfg.remove_mentions:
        text = MENTION_RE.sub(" ", text)
    if cfg.remove_repeated_punctuation:
        text = REPEAT_PUNCT_RE.sub(r"\1\1", text)
    if cfg.max_repeat_char and cfg.max_repeat_char > 0:
        text = REPEAT_CHAR_RE.sub(lambda m: m.group(1) * cfg.max_repeat_char, text)

    # 5. whitespace / newline / tab normalisation --------------------------
    if cfg.normalize_unicode_spaces:
        text = ALL_SPACE_RE.sub(" ", text)
    if cfg.fix_punct_spacing:
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    if cfg.collapse_to_single_space:
        text = ALL_SPACE_RE.sub(" ", text)
    if cfg.strip_edges:
        text = text.strip()

    # 6. validity ----------------------------------------------------------
    # Only rows that end up with *no visible character at all* are dropped
    # (null / NaN / whitespace-only / zero-width-only).  Emoji-only or
    # punctuation-only comments are real ViHSD samples and MUST be kept:
    # the plan forbids removing test samples just because of preprocessing.
    if not text:
        return None
    if _visible_length(text) < cfg.min_chars:
        return None
    return text


def clean_series(series: pd.Series, cfg: CleaningConfig | None = None) -> pd.Series:
    """Vectorised-friendly cleaning that preserves the original index."""
    cfg = cfg or CleaningConfig()
    return series.apply(lambda v: clean_text(v, cfg))


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    cfg: CleaningConfig | None = None,
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Clean ``df[text_column]`` in place-ish and report how many rows were dropped.

    Rows are *never* dropped because of the label: only rows without usable text
    are removed (plan section 3).
    """
    cfg = cfg or CleaningConfig()
    stats = {"input_rows": int(len(df))}

    if text_column not in df.columns:
        raise KeyError(f"Text column {text_column!r} not found in columns {list(df.columns)}")

    out = df.copy()
    cleaned = clean_series(out[text_column], cfg)

    invalid_mask = cleaned.isna()
    stats["rows_without_usable_text"] = int(invalid_mask.sum())

    if drop_invalid:
        out = out.loc[~invalid_mask].copy()
        cleaned = cleaned.loc[~invalid_mask]
    else:
        cleaned = cleaned.fillna("")

    out[text_column] = cleaned.astype(str).values
    stats["output_rows"] = int(len(out))
    return out.reset_index(drop=True), stats


def build_cleaning_config(overrides: Dict[str, object] | None = None) -> CleaningConfig:
    """Build a :class:`CleaningConfig`, ignoring unknown keys."""
    cfg = CleaningConfig()
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key) and not key.startswith("_"):
            setattr(cfg, key, value)
    return cfg


def preview_examples(df: pd.DataFrame, text_column: str = "text", n: int = 5) -> List[Dict[str, str]]:
    """Small before/after sample used for the preprocessing report."""
    sample = df.head(n)
    return [{"before": str(row), "after": str(sample[text_column].iloc[i])}
            for i, row in enumerate(sample[text_column])]
