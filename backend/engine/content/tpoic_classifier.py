"""Topic classification for the TrustLens AI Content Understanding Engine.

Single responsibility: given cleaned text, identify the primary topic
and any relevant secondary topics it discusses, each with a
confidence score. Uses a Hugging Face Transformers zero-shot
classification pipeline against a fixed candidate topic taxonomy, so
no task-specific fine-tuning or training data is required. Performs
no summarization or other text generation.
"""

import logging

from transformers import Pipeline, pipeline

from engines.content.schemas import TopicClassificationResult, TopicScore

logger = logging.getLogger(__name__)

_ZERO_SHOT_MODEL_NAME: str = "facebook/bart-large-mnli"
_MAX_INPUT_CHARACTERS: int = 2000
_DEFAULT_SECONDARY_TOPIC_COUNT: int = 3
_DEFAULT_MIN_CONFIDENCE: float = 0.15

_CANDIDATE_TOPICS: list[str] = [
    "Politics",
    "Technology",
    "Health",
    "Science",
    "Business",
    "Finance",
    "Sports",
    "Entertainment",
    "Environment",
    "Education",
    "Crime",
    "World News",
    "Opinion",
    "Lifestyle",
]

_classifier: Pipeline | None = None


class TopicClassificationError(Exception):
    """Raised when topic classification fails."""


def _get_classifier() -> Pipeline:
    """Return a lazily initialized, process-wide zero-shot classification pipeline.

    Loading the underlying transformer model is expensive, so the
    pipeline is constructed once per process and reused across calls
    rather than being reloaded on every invocation.

    Returns:
        Pipeline: The shared zero-shot classification pipeline.
    """
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot classification model: %s", _ZERO_SHOT_MODEL_NAME)
        _classifier = pipeline("zero-shot-classification", model=_ZERO_SHOT_MODEL_NAME)
    return _classifier


def classify_topics(
    text: str,
    candidate_labels: list[str] | None = None,
    secondary_topic_count: int = _DEFAULT_SECONDARY_TOPIC_COUNT,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> TopicClassificationResult:
    """Identify the primary and secondary topics discussed in text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` after language detection. Text is truncated to a
    bounded character length before classification, both for latency
    and to stay within the underlying model's input length limits.

    Args:
        text: The cleaned text to classify.
        candidate_labels: The topic taxonomy to classify against. If
            ``None`` (the default), a built-in general-purpose topic
            taxonomy is used.
        secondary_topic_count: Maximum number of secondary topics to
            return, beyond the primary topic. Defaults to 3.
        min_confidence: Minimum confidence score a secondary topic
            must meet to be included. Defaults to 0.15.

    Raises:
        TopicClassificationError: If ``text`` is empty/whitespace-only,
            or the underlying classification model fails.

    Returns:
        TopicClassificationResult: The primary topic and its
            confidence, plus a list of qualifying secondary topics
            with their confidence scores.
    """
    if not text or not text.strip():
        raise TopicClassificationError("Cannot classify topics of empty text.")

    labels = candidate_labels or _CANDIDATE_TOPICS
    truncated_text = text[:_MAX_INPUT_CHARACTERS]

    try:
        classifier = _get_classifier()
        output = classifier(truncated_text, candidate_labels=labels, multi_label=True)
    except Exception as exc:
        raise TopicClassificationError(f"Topic classification failed: {exc}") from exc

    ranked_labels: list[str] = output["labels"]
    ranked_scores: list[float] = output["scores"]

    if not ranked_labels:
        raise TopicClassificationError("Topic classification returned no results.")

    primary_topic = ranked_labels[0]
    primary_confidence = round(ranked_scores[0], 4)

    secondary_topics = [
        TopicScore(topic=label, confidence=round(score, 4))
        for label, score in zip(ranked_labels[1:], ranked_scores[1:], strict=False)
        if score >= min_confidence
    ][:secondary_topic_count]

    result = TopicClassificationResult(
        primary_topic=primary_topic,
        primary_confidence=primary_confidence,
        secondary_topics=secondary_topics,
    )

    logger.info(
        "Classified topics: primary=%s (%.4f), secondary=%s",
        result.primary_topic,
        result.primary_confidence,
        [s.topic for s in result.secondary_topics],
    )
    return result