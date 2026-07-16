"""Language detection for the TrustLens AI Content Understanding Engine.

Single responsibility: given cleaned text, detect its dominant
language and report a confidence score. spaCy does not ship reliable
built-in language identification, so this module uses ``langdetect``
(a pure-Python port of Google's language-detection library) \u2014 the
appropriate, project-compatible approach flagged during tech-stack
selection. Performs no further NLP; downstream stages own that.
"""

import logging

from langdetect import DetectorFactory, LangDetectException, detect_langs

from engines.content.schemas import LanguageDetectionResult

logger = logging.getLogger(__name__)

# Fixes langdetect's internal random seed so repeated calls on the same
# text deterministically return the same result.
DetectorFactory.seed = 0

_MIN_TEXT_LENGTH_FOR_RELIABLE_DETECTION = 20


class LanguageDetectionError(Exception):
    """Raised when language cannot be detected from the given text."""


def detect_language(text: str) -> LanguageDetectionResult:
    """Detect the dominant language of a block of text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` after text cleaning.

    Args:
        text: The cleaned text to detect the language of.

    Raises:
        LanguageDetectionError: If ``text`` is empty/whitespace-only,
            or no language could be confidently detected (e.g. text
            is too short, numeric-only, or contains no linguistic
            content).

    Returns:
        LanguageDetectionResult: The detected ISO 639-1 language code
            and the detector's confidence score (between 0.0 and 1.0).
    """
    if not text or not text.strip():
        raise LanguageDetectionError("Cannot detect language of empty text.")

    if len(text.strip()) < _MIN_TEXT_LENGTH_FOR_RELIABLE_DETECTION:
        logger.warning(
            "Text is shorter than %d characters; language detection may be unreliable.",
            _MIN_TEXT_LENGTH_FOR_RELIABLE_DETECTION,
        )

    try:
        candidates = detect_langs(text)
    except LangDetectException as exc:
        raise LanguageDetectionError(f"Language detection failed: {exc}") from exc

    if not candidates:
        raise LanguageDetectionError("Language detection returned no candidates.")

    top_candidate = candidates[0]
    result = LanguageDetectionResult(
        language=top_candidate.lang,
        confidence=round(top_candidate.prob, 4),
    )

    logger.info("Detected language=%s confidence=%.4f", result.language, result.confidence)
    return result