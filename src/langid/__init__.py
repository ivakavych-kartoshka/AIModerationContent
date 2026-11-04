"""Language identification, tokenization and routing for SensitiveAI.

The package is intentionally standalone: it does not touch training or
evaluation.  It sits before the inference/fusion stage and answers three
questions:

    which language is this comment?          -> LanguageDetector
    which tokens/spans belong to which one?  -> LanguageAwareTokenizer
    which rule engine / model should scan it? -> LanguageRouter
"""

from .detector import (
    LANGUAGES,
    DetectionConfig,
    LanguageDetector,
    LanguageResult,
    blend_scores,
)
from .heuristics import (
    EN_CONTENT_WORDS,
    EN_WORDS,
    VI_CHARS,
    VI_SPECIFIC_CHARS,
    VI_TONED_CHARS,
    VI_WORDS,
    HeuristicSignal,
    heuristic_signal,
    token_language,
    word_tokens,
)
from .tokenizer import LangToken, LanguageAwareTokenizer, LanguageSegment

__all__ = [
    "EN_CONTENT_WORDS",
    "EN_WORDS",
    "LANGUAGES",
    "VI_CHARS",
    "VI_SPECIFIC_CHARS",
    "VI_TONED_CHARS",
    "VI_WORDS",
    "DetectionConfig",
    "HeuristicSignal",
    "LangToken",
    "LanguageAwareTokenizer",
    "LanguageDetector",
    "LanguageResult",
    "LanguageSegment",
    "blend_scores",
    "heuristic_signal",
    "token_language",
    "word_tokens",
]
