"""Tokenize text and tag every token with its language.

This is the "tokenize for language routing" layer.  It does not replace the
model tokenizer (``mDeBERTa`` keeps doing that); it produces language-tagged
spans so the router can:

* run the Vietnamese lexicon on Vietnamese spans,
* run the English lexicon on English spans,
* keep a code-mixed comment's English profanity even when the comment is mostly
  Vietnamese.

Segments are built from word offsets, so each segment keeps its exact position
in the original text - the same offset discipline the rule engine already uses
for its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..common.logging_utils import get_logger
from .detector import LanguageDetector, LanguageResult
from .heuristics import (
    EN_CONTENT_WORDS,
    EN_WORDS,
    VI_CHARS,
    VI_WORDS,
    normalize_unicode,
    split_words_with_offsets,
)

LOGGER = get_logger("sensitiveai.langid.tokenizer")


@dataclass
class LangToken:
    """One word token with a language tag and its source offsets."""

    text: str
    start: int
    end: int
    lang: str = "unknown"  # vi | en | unknown
    strong: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "lang": self.lang,
            "strong": self.strong,
        }


@dataclass
class LanguageSegment:
    """A contiguous span of tokens that share a language."""

    text: str
    start: int
    end: int
    lang: str
    tokens: List[LangToken] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "lang": self.lang,
            "n_tokens": self.n_tokens,
            "tokens": [token.to_dict() for token in self.tokens],
        }


class LanguageAwareTokenizer:
    """Word tokenizer + per-token language tags + contiguous segments."""

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        min_segment_tokens: int = 3,
    ) -> None:
        self.detector = detector or LanguageDetector()
        self.min_segment_tokens = max(1, int(min_segment_tokens))

    # ------------------------------------------------------------------ #
    # token tagging
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tag(token: str) -> tuple[str, bool]:
        """Language and strength of a single word.

        Strong tokens are real evidence (Vietnamese diacritics / Vietnamese
        vocabulary, or an English *content* word).  Common English function
        words are tagged English but left weak so a stray ``the`` does not split
        a Vietnamese comment into a separate segment.
        """
        lower = normalize_unicode(token).casefold()
        if any(char in VI_CHARS for char in lower):
            return "vi", True
        if lower in EN_CONTENT_WORDS:
            return "en", True
        if lower in EN_WORDS:
            return "en", False
        if lower in VI_WORDS:
            return "vi", True
        return "unknown", False

    def tokenize(self, text: object) -> tuple[List[LangToken], LanguageResult]:
        """Tokenize ``text`` and fill unknown tags from context."""
        raw = "" if text is None else str(text)
        doc = self.detector.detect(raw)
        tokens = [
            LangToken(text=word, start=start, end=end, lang=lang, strong=strong)
            for word, start, end in split_words_with_offsets(raw)
            for lang, strong in [self._tag(word)]
        ]

        # Unknown, non-strong tokens inherit the document language.  This is
        # safe because an unknown word is by definition not a lexicon term for
        # either language, so it does not change which lexicon should scan it.
        fallback = doc.lang if doc.lang in ("vi", "en") else "unknown"
        for token in tokens:
            if token.lang == "unknown":
                token.lang = fallback

        # Any remaining unknown (document also unknown) inherits the previous
        # known tag, then the next one, so spans stay contiguous.
        for index, token in enumerate(tokens):
            if token.lang != "unknown":
                continue
            previous = next((t.lang for t in reversed(tokens[:index]) if t.lang != "unknown"), None)
            following = next((t.lang for t in tokens[index + 1 :] if t.lang != "unknown"), None)
            token.lang = previous or following or "unknown"
        return tokens, doc

    # ------------------------------------------------------------------ #
    # segmentation
    # ------------------------------------------------------------------ #
    def _segments_from_tokens(self, raw: str, tokens: Sequence[LangToken]) -> List[LanguageSegment]:
        if not tokens:
            return []
        groups: List[List[LangToken]] = []
        current: List[LangToken] = [tokens[0]]
        for token in tokens[1:]:
            if token.lang == current[-1].lang:
                current.append(token)
            else:
                groups.append(current)
                current = [token]
        groups.append(current)

        # Absorb tiny, weak groups into the nearest larger neighbour.  A tiny
        # *strong* group is kept: one English content word inside a Vietnamese
        # comment is exactly what per-segment scanning exists to catch.
        changed = True
        while changed and len(groups) > 1:
            changed = False
            for index, group in enumerate(groups):
                if len(group) >= self.min_segment_tokens:
                    continue
                if any(token.strong for token in group):
                    continue
                neighbours = []
                if index > 0:
                    neighbours.append(index - 1)
                if index + 1 < len(groups):
                    neighbours.append(index + 1)
                if not neighbours:
                    continue
                target = max(neighbours, key=lambda i: len(groups[i]))
                groups[target].extend(group)
                groups[target].sort(key=lambda token: token.start)
                del groups[index]
                changed = True
                break

        # Re-merge adjacent groups that now share a language.
        merged: List[List[LangToken]] = []
        for group in groups:
            if merged and merged[-1][0].lang == group[0].lang:
                merged[-1].extend(group)
            else:
                merged.append(group)

        segments: List[LanguageSegment] = []
        for group in merged:
            start = group[0].start
            end = group[-1].end
            votes: Dict[str, int] = {}
            for token in group:
                votes[token.lang] = votes.get(token.lang, 0) + 1
            lang = max(votes, key=lambda key: votes[key]) if votes else "unknown"
            segments.append(
                LanguageSegment(
                    text=raw[start:end],
                    start=start,
                    end=end,
                    lang=lang,
                    tokens=list(group),
                )
            )
        return segments

    def segment(self, text: object) -> tuple[List[LanguageSegment], LanguageResult]:
        """Return ``(segments, document language result)``."""
        raw = "" if text is None else str(text)
        tokens, doc = self.tokenize(raw)
        return self._segments_from_tokens(raw, tokens), doc

    def dominant_language(self, text: object) -> LanguageResult:
        return self.detector.detect(text)


__all__ = [
    "LangToken",
    "LanguageAwareTokenizer",
    "LanguageSegment",
]
