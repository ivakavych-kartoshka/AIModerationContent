from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import logging
import unicodedata
from pathlib import Path

app = FastAPI(title="SensitiveAI Moderation Service", version="1.1.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sensitiveai-api")

# ---------------------------------------------------------------------------
# Hybrid moderation: Rule Engine + language detection
# ---------------------------------------------------------------------------
try:
    from src.langid.detector import LanguageDetector
    from src.routing.language_router import LanguageRouter

    _detector = LanguageDetector()
    _router = LanguageRouter.from_config_files()
    HYBRID_AVAILABLE = True
except Exception as exc:  # pragma: no cover - defensive fallback
    logger.warning("Hybrid system unavailable, falling back to keyword heuristic: %s", exc)
    _detector = None
    _router = None
    HYBRID_AVAILABLE = False


def _looks_vietnamese(text: str) -> bool:
    """Dependency-free fallback: Vietnamese text carries diacritics."""
    for char in text:
        if char in "ăâđêôơưĂÂĐÊÔƠƯ":
            return True
        if unicodedata.combining(char):
            return True
    return False


def detect_language(text: str) -> str:
    """Return one of ``vi``, ``en``, ``mix``, ``other``."""
    if _detector is not None:
        return _detector.detect(text).lang
    return "vi" if _looks_vietnamese(text) else "en"


def _message_lang(lang: str) -> str:
    """Pick the message language for a detection verdict."""
    if lang in ("vi", "en"):
        return lang
    if lang == "mix":
        return "vi"
    return "en"


# ---------------------------------------------------------------------------
# Localized notification messages
# ---------------------------------------------------------------------------
DECISION_MESSAGES = {
    "BLOCK": {
        "vi": "Nội dung của bạn chứa từ ngữ không phù hợp nên đã bị chặn.",
        "en": "Your content contains inappropriate language and has been blocked.",
    },
    "REVIEW": {
        "vi": "Nội dung của bạn đang chờ kiểm duyệt viên xem xét.",
        "en": "Your content is pending review by a moderator.",
    },
    "ALLOW": {
        "vi": "Nội dung của bạn đã được duyệt và hiển thị công khai.",
        "en": "Your content has been approved and is now public.",
    },
}

RECOMMENDATIONS = {
    "BLOCK": {
        "vi": "Vui lòng chỉnh sửa nội dung và đăng lại sau khi loại bỏ từ ngữ vi phạm.",
        "en": "Please edit the content and resubmit after removing the violating language.",
    },
    "REVIEW": {
        "vi": "Vui lòng chờ trong giây lát, nội dung sẽ hiển thị sau khi được duyệt.",
        "en": "Please wait; the content will appear once it has been approved.",
    },
    "ALLOW": {
        "vi": "Không cần thao tác thêm.",
        "en": "No further action is required.",
    },
}

RISK_BY_DECISION = {"BLOCK": "HIGH", "REVIEW": "MEDIUM", "ALLOW": "LOW"}

# Combined bilingual wording for code-mixed content.
SEPARATOR = " / "


def _localized(table: dict, decision: str, lang: str) -> str:
    """Return a localized sentence, bilingual for ``mix``."""
    key = decision.upper()
    messages = table[key]
    if lang == "mix":
        return messages["vi"] + SEPARATOR + messages["en"]
    return messages[_message_lang(lang)]


class ModerationRequest(BaseModel):
    text: str
    context: Optional[str] = None


class BatchModerationRequest(BaseModel):
    items: List[ModerationRequest]


class ModerationResponse(BaseModel):
    decision: str  # BLOCK, REVIEW, ALLOW
    score: float
    categories: List[str]
    matches: Optional[List[Any]] = None
    risk: Optional[str] = None
    recommendation: Optional[str] = None
    language: Optional[str] = None
    message: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "sensitiveai-moderation"}


@app.post("/api/v1/moderate", response_model=ModerationResponse)
def moderate_content(request: ModerationRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Nội dung không được để trống / Content must not be empty.",
        )

    text_lower = request.text.lower()
    language = detect_language(request.text)

    # Try hybrid system (rule engine + language routing)
    if HYBRID_AVAILABLE:
        try:
            routed = _router.route(request.text)
            hate_score = float(routed.rule_hate_score)
            offensive_score = float(routed.rule_offensive_score)
            
            # Decision thresholds: high = BLOCK, medium = REVIEW, low = ALLOW
            max_score = max(hate_score, offensive_score)
            
            if max_score >= 0.5:
                decision = "BLOCK"
                score = min(0.95, max_score + 0.25)
                categories = []
                if hate_score >= 0.3:
                    categories.append("HATE")
                if offensive_score >= 0.3:
                    categories.append("OFFENSIVE")
                matches = routed.matches if routed.matches else []
            elif max_score >= 0.3:
                decision = "REVIEW"
                score = max_score
                categories = ["SUSPICIOUS"]
                matches = []
            else:
                decision = "ALLOW"
                score = 1.0 - max_score
                categories = []
                matches = []
        except Exception as e:
            logger.warning("Rule engine failed, falling back to keyword heuristic: %s", e)
            decision, score, categories, matches = _fallback_heuristic(text_lower)
    else:
        # Fallback to simple keyword heuristic
        decision, score, categories, matches = _fallback_heuristic(text_lower)

    return ModerationResponse(
        decision=decision,
        score=score,
        categories=categories,
        matches=matches,
        risk=RISK_BY_DECISION[decision],
        recommendation=_localized(RECOMMENDATIONS, decision, language),
        language=language,
        message=_localized(DECISION_MESSAGES, decision, language),
    )


def _fallback_heuristic(text_lower: str):
    """Fallback keyword-based moderation when rule engine unavailable."""
    toxic_keywords = [
        # Vietnamese
        "xấu", "tệ", "ngu", "đần", "dm", "vl", "đm", "cút", "lừa đảo",
        # English
        "fuck", "shit", "bitch", "asshole", "bastard", "damn", "cunt", "dick", "piss", "hell",
        "motherfucker", "fucker", "bullshit", "horseshit", "crap", "wtf", "stfu", "gtfo",
        # Common Vietnamese-English mixed
        "fucking", "shitting", "bitching",
    ]
    found = [kw for kw in toxic_keywords if kw in text_lower]

    if len(found) > 0:
        return "BLOCK", 0.95, ["TOXIC", "HATE"], found
    elif "?" in text_lower and len(text_lower) > 50:
        return "REVIEW", 0.6, ["SUSPICIOUS"], []
    else:
        return "ALLOW", 0.05, [], []


@app.post("/api/v1/moderate/batch", response_model=List[ModerationResponse])
def moderate_batch(request: BatchModerationRequest):
    return [moderate_content(item) for item in request.items]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
