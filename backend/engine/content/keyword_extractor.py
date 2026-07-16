"""Keyword extraction for the TrustLens AI Content Understanding Engine.

Single responsibility: given cleaned text, extract meaningful
keywords and keyphrases (noun chunks and proper/common noun tokens),
strip stop words and determiners, deduplicate case-insensitively, and
rank the result by frequency. Performs no summarization, topic
modeling, or entity classification; downstream stages own that.
"""

import logging
from collections import Counter

import spacy
from spacy.language import Language
from spacy.tokens import Span, Token

from engines.content.schemas import Keyword

logger = logging.getLogger(__name__)

_SPACY_MODEL_NAME: str = "en_core_web_sm"
_MAX_INPUT_CHARACTERS: int = 100_000
_DEFAULT_TOP_N: int = 15
_MIN_KEYWORD_LENGTH: int = 2
_LEADING_STRIP_POS: frozenset[str] = frozenset({"DET", "PRON", "ADP"})

_nlp: Language | None = None


class KeywordExtractionError(Exception):
    """Raised when keyword extraction fails."""


def _get_nlp() -> Language:
    """Return a lazily initialized, process-wide spaCy language pipeline.

    Loading a spaCy model is relatively expensive, so it is loaded
    once per process and reused across calls rather than being
    reloaded on every invocation.

    Returns:
        Language: The shared spaCy pipeline instance.
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model: %s", _SPACY_MODEL_NAME)
        _nlp = spacy.load(_SPACY_MODEL_NAME)
    return _nlp


def _normalize_chunk(chunk: Span) -> str | None:
    """Normalize a noun chunk into a clean keyword candidate string.

    Strips leading determiners, pronouns, and prepositions (e.g.
    turns "the global economy" into "global economy"), then joins the
    remaining tokens' lowercased text.

    Args:
        chunk: A spaCy noun-chunk span.

    Returns:
        str | None: The normalized keyword text, or ``None`` if
            nothing meaningful remains after stripping.
    """
    tokens = list(chunk)
    while tokens and tokens[0].pos_ in _LEADING_STRIP_POS:
        tokens = tokens[1:]

    meaningful_tokens = [
        token.text.lower()
        for token in tokens
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]

    candidate = " ".join(meaningful_tokens).strip()
    return candidate if len(candidate) >= _MIN_KEYWORD_LENGTH else None


def _is_meaningful_token(token: Token) -> bool:
    """Determine whether a single token is a viable standalone keyword.

    Args:
        token: A spaCy token.

    Returns:
        bool: ``True`` if the token is a non-stopword, non-punctuation
            noun or proper noun of sufficient length.
    """
    return (
        token.pos_ in ("NOUN", "PROPN")
        and not token.is_stop
        and not token.is_punct
        and len(token.text) >= _MIN_KEYWORD_LENGTH
    )


def extract_keywords(text: str, top_n: int = _DEFAULT_TOP_N) -> list[Keyword]:
    """Extract, deduplicate, and rank keywords from text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` after language detection. Candidate keywords are
    drawn from noun chunks (multi-word phrases with determiners
    stripped) and standalone noun/proper-noun tokens, deduplicated
    case-insensitively, and ranked by frequency of occurrence.

    Args:
        text: The cleaned text to extract keywords from.
        top_n: Maximum number of ranked keywords to return. Defaults
            to 15.

    Raises:
        KeywordExtractionError: If ``text`` is empty/whitespace-only.

    Returns:
        list[Keyword]: Keywords ordered by descending relevance score,
            each with a normalized score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        raise KeywordExtractionError("Cannot extract keywords from empty text.")

    nlp = _get_nlp()
    doc = nlp(text[:_MAX_INPUT_CHARACTERS])

    candidate_counts: Counter[str] = Counter()
    display_forms: dict[str, str] = {}

    for chunk in doc.noun_chunks:
        normalized = _normalize_chunk(chunk)
        if normalized:
            candidate_counts[normalized] += 1
            display_forms.setdefault(normalized, normalized)

    for token in doc:
        if _is_meaningful_token(token):
            normalized = token.text.lower()
            candidate_counts[normalized] += 1
            display_forms.setdefault(normalized, normalized)

    if not candidate_counts:
        logger.warning("No keyword candidates found in text.")
        return []

    max_count = max(candidate_counts.values())
    ranked = sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))

    keywords = [
        Keyword(text=display_forms[normalized], score=round(count / max_count, 4))
        for normalized, count in ranked[:top_n]
    ]

    logger.info("Extracted %d keyword(s) from text (top: %s).", len(keywords), keywords[0].text if keywords else None)
    return keywords