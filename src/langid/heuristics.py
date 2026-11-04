"""Cheap, dependency-free language signals for Vietnamese vs English.

Everything here is a *heuristic*: it says how much evidence a text carries for
Vietnamese and for English.  It never makes the final decision on its own - the
detector combines these signals with an optional statistical backend
(``langid``) and with explicit code-mix rules.

Why a separate module: the signals are also used by
:class:`src.langid.tokenizer.LanguageAwareTokenizer` to tag **individual
tokens**, which is what makes per-segment routing (and per-segment lexicon
scanning) possible.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Vietnamese character classes
# --------------------------------------------------------------------------- #
#: Letters that exist in Vietnamese but not in plain English.  These are the
#: strongest single-character signal (``đ`` is impossible in English, ``ă/â/ê/
#: ô/ơ/ư`` are Vietnamese-specific).
VI_SPECIFIC_CHARS = set("ăâđêôơưĂÂĐÊÔƠƯ")

#: Tonal vowels.  Also used by Portuguese/Spanish/etc., so they are weaker
#: evidence than :data:`VI_SPECIFIC_CHARS`.
VI_TONED_CHARS = set(
    "àáảãạ"
    "ằắẳẵặ"
    "ầấẩẫậ"
    "èéẻẽẹ"
    "ềếểễệ"
    "ìíỉĩị"
    "òóỏõọ"
    "ồốổỗộ"
    "ờớởỡợ"
    "ùúủũụ"
    "ừứửữự"
    "ỳýỷỹỵ"
    "ÀÁẢÃẠ"
    "ẰẮẲẴẶ"
    "ẦẤẨẪẬ"
    "ÈÉẺẼẸ"
    "ỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌ"
    "ỒỐỔỖỘ"
    "ỜỚỞỠỢ"
    "ÙÚỦŨỤ"
    "ỪỨỬỮỰ"
    "ỲÝỶỸỴ"
)

#: All characters that mark a Vietnamese-looking token.
VI_CHARS = VI_SPECIFIC_CHARS | VI_TONED_CHARS

#: Word regex: unicode letters with optional inner apostrophe/hyphen.
WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*", re.UNICODE)

# --------------------------------------------------------------------------- #
# Word lists
# --------------------------------------------------------------------------- #
#: Very frequent Vietnamese function/content words.  Includes the common
#: unaccented spellings so that ``"may noi the la sai"`` is still recognised.
VI_WORDS = frozenset(
    """
    và là của có không khong được duoc cho với voi này nay đó do thì thi mà ma
    như nhu để de ra vào vao một mot những nhung các cac người nguoi ta tôi toi
    mình minh bạn ban anh em chị chi ông ong bà ba cô co chú chu thằng thang con
    nó no họ ho chúng chung ai gì gi sao nào nao đâu dau rồi roi đã da đang
    cũng cung vẫn van rất rat quá qua lắm lam thế the vậy vay nếu neu khi khi
    trong ngoài ngoai trên tren dưới duoi trước truoc sau về ve từ tu đến den
    tại tai vì vi bởi boi nên nen nhưng nhung hoặc hoac hay với voi cùng cung
    thật that thực thuc đúng dung sai tốt tot xấu xau ghê ghe vãi vai vl vcl
    vlon đéo deo đm dm đmm dmm đcm dcm cặc cac lồn lon đĩ di chó cho khốn khon
    nạn nan súc suc vật vat rác rac não nao ngu ngốc ngoc đần dan khùng khung
    điên dien mẹ me cha bố bo ông ong kệ ke thôi thoi nhé nhe nha nhá nha ạ a
    ừ u uh ok được duoc rồi roi đấy day ấy ay
    """.split()
)

#: Frequent English words.  Content words and function words are kept together
#: because both are useful evidence for short comments.
EN_WORDS = frozenset(
    """
    the and is are was were be been being am do does did have has had to of in
    on at for with from by about as into through during before after above below
    between out up down off over under again further then once here there when
    where why how all any both each few more most other some such no nor not only
    own same so than too very can will just should now i me my we our us you your
    he him his she her it its they them their this that these those what which
    who whom whose if because while until although though since unless
    idiot stupid dumb moron dumbass retard retarded fool jerk loser pathetic
    trash garbage scum worthless useless hate hated hateful kill die death
    fuck fucking shit bitch bastard asshole dick cunt slut whore faggot nigger
    nigga kike spic chink tranny gay queer rape rapist nazi terrorist
    """.split()
)

#: English words that are strong signals on their own.  They are the kind of
#: term a Vietnamese comment mixes in (often profanity), so one occurrence is
#: enough to make a comment "code-mixed" rather than pure Vietnamese.
EN_CONTENT_WORDS = frozenset(
    """
    idiot stupid dumb moron dumbass retard retarded fool jerk loser pathetic
    trash garbage scum worthless useless hate hateful kill die death fuck
    fucking shit bitch bastard asshole dick cunt slut whore faggot nigger nigga
    kike spic chink tranny gay queer rape rapist nazi terrorist
    """.split()
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def word_tokens(text: str) -> List[str]:
    """Lower-cased word tokens (punctuation/digits dropped)."""
    return [match.group(0).casefold() for match in WORD_RE.finditer(text or "")]


def character_profile(text: str) -> Dict[str, float]:
    """Character-level Vietnamese diacritic ratios."""
    normalized = normalize_unicode(text or "")
    alpha = 0
    specific = 0
    toned = 0
    for char in normalized:
        if not char.isalpha():
            continue
        alpha += 1
        if char in VI_SPECIFIC_CHARS:
            specific += 1
        elif char in VI_TONED_CHARS:
            toned += 1
    if alpha == 0:
        return {"alpha_chars": 0.0, "specific_ratio": 0.0, "toned_ratio": 0.0}
    return {
        "alpha_chars": float(alpha),
        "specific_ratio": specific / alpha,
        "toned_ratio": toned / alpha,
    }


@dataclass(frozen=True)
class HeuristicSignal:
    """Evidence for one text, all values in ``[0, 1]`` except counts."""

    vi_score: float
    en_score: float
    n_words: int
    vi_word_ratio: float
    en_word_ratio: float
    en_content_ratio: float
    specific_ratio: float
    toned_ratio: float
    has_vi_diacritics: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "vi_score": round(self.vi_score, 4),
            "en_score": round(self.en_score, 4),
            "n_words": self.n_words,
            "vi_word_ratio": round(self.vi_word_ratio, 4),
            "en_word_ratio": round(self.en_word_ratio, 4),
            "en_content_ratio": round(self.en_content_ratio, 4),
            "specific_ratio": round(self.specific_ratio, 4),
            "toned_ratio": round(self.toned_ratio, 4),
            "has_vi_diacritics": self.has_vi_diacritics,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def token_language(token: str) -> str:
    """Best guess for a single token: ``vi``, ``en`` or ``unknown``.

    Single-token decisions are intentionally conservative.  Vietnamese
    diacritics are decisive; otherwise English words win over unaccented
    Vietnamese look-alikes (``the``, ``no``, ``do``) so that English sentences
    are not split into Vietnamese segments.
    """
    token = normalize_unicode(token).casefold()
    if not token:
        return "unknown"
    if any(char in VI_CHARS for char in token):
        return "vi"
    if token in EN_CONTENT_WORDS:
        return "en"
    if token in EN_WORDS:
        return "en"
    if token in VI_WORDS:
        return "vi"
    return "unknown"


def heuristic_signal(text: str) -> HeuristicSignal:
    """Compute the dependency-free Vietnamese/English evidence for ``text``."""
    text = text or ""
    profile = character_profile(text)
    tokens = word_tokens(text)
    n_words = len(tokens)
    if n_words == 0:
        return HeuristicSignal(
            vi_score=0.0,
            en_score=0.0,
            n_words=0,
            vi_word_ratio=0.0,
            en_word_ratio=0.0,
            en_content_ratio=0.0,
            specific_ratio=profile["specific_ratio"],
            toned_ratio=profile["toned_ratio"],
            has_vi_diacritics=False,
        )

    vi_hits = 0
    en_hits = 0
    en_content_hits = 0
    for token in tokens:
        if any(char in VI_CHARS for char in token) or token in VI_WORDS:
            vi_hits += 1
        if token in EN_WORDS:
            en_hits += 1
        if token in EN_CONTENT_WORDS:
            en_content_hits += 1

    vi_word_ratio = vi_hits / n_words
    en_word_ratio = en_hits / n_words
    en_content_ratio = en_content_hits / n_words

    # A Vietnamese sentence with diacritics typically has >8% accented letters;
    # ``specific_ratio`` is weighted up because those letters are unambiguous.
    diacritic_signal = _clamp(
        (0.60 * profile["toned_ratio"] + 1.60 * profile["specific_ratio"]) / 0.12
    )
    vi_word_signal = _clamp(vi_word_ratio * 2.2)
    en_word_signal = _clamp(en_word_ratio * 2.2)

    # Word evidence is worth more than diacritics: "café" has an accent but no
    # Vietnamese word, while "may noi the" has no accent but three VI words.
    vi_score = _clamp(0.38 * diacritic_signal + 0.62 * vi_word_signal)
    en_score = _clamp(0.85 * en_word_signal + 0.15 * _clamp(en_content_ratio * 3.0))

    return HeuristicSignal(
        vi_score=vi_score,
        en_score=en_score,
        n_words=n_words,
        vi_word_ratio=vi_word_ratio,
        en_word_ratio=en_word_ratio,
        en_content_ratio=en_content_ratio,
        specific_ratio=profile["specific_ratio"],
        toned_ratio=profile["toned_ratio"],
        has_vi_diacritics=bool(profile["specific_ratio"] or profile["toned_ratio"]),
    )


def split_words_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Return ``(word, start, end)`` for every word-like token in ``text``."""
    text = text or ""
    return [(match.group(0), match.start(), match.end()) for match in WORD_RE.finditer(text)]


__all__ = [
    "EN_CONTENT_WORDS",
    "EN_WORDS",
    "HeuristicSignal",
    "VI_CHARS",
    "VI_SPECIFIC_CHARS",
    "VI_TONED_CHARS",
    "VI_WORDS",
    "WORD_RE",
    "character_profile",
    "heuristic_signal",
    "normalize_unicode",
    "split_words_with_offsets",
    "token_language",
    "word_tokens",
]
