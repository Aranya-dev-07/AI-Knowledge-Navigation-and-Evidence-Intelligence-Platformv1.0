"""Document classification for the TrustLens AI Content Understanding Engine.

Single responsibility: given cleaned text, classify the overall
document into one of a fixed set of content-type categories (News,
Research Paper, Government Report, Blog, Social Media, Documentation,
Legal, Medical, Educational, Other), with a confidence score. Uses a
Hugging Face Transformers zero-shot classification pipeline, so no
task-specific fine-tuning is required. This is the final analytical
stage before the pipeline assembles its structured JSON output.
"""

import logging

from transformers import Pipeline, pipeline

from engines.content.schemas import ContentClassification

logger = logging.getLogger(__name__)

_ZERO_SHOT_MODEL_NAME: str = "facebook/bart-large-mnli"
_MAX_INPUT_CHARACTERS: int = 2000

_CONTENT_CLASSES: list[str] = [
    "News",
    "Research Paper",
    "Government Report",
    "Blog",
    "Social Media",
    "Documentation",
    "Legal",
    "Medical",
    "Educational",
    "Other",
]

_classifier: Pipeline | None = None


class ContentClassificationError(Exception):
    """Raised when document classification fails."""


def _get_classifier() -> Pipeline:
    """Return a lazily initialized, process-wide zero-shot classification pipeline.

    Maintained separately from ``topic_classifier``'s and
    ``claim_extractor``'s pipeline instances to keep this module
    independently usable, at the cost of caching the same underlying
    model multiple times in memory if several such modules are used
    together.

    Returns:
        Pipeline: The shared zero-shot classification pipeline.
    """
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot classification model: %s", _ZERO_SHOT_MODEL_NAME)
        _classifier = pipeline("zero-shot-classification", model=_ZERO_SHOT_MODEL_NAME)
    return _classifier


def classify_content(text: str) -> ContentClassification:
    """Classify the overall document type of a piece of text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` as the final analytical stage before structured
    JSON output assembly.

    Args:
        text: The cleaned text to classify.

    Raises:
        ContentClassificationError: If ``text`` is empty/whitespace-only,
            or the underlying classification model fails.

    Returns:
        ContentClassification: The single best-matching document
            category and the classifier's confidence score.
    """
    if not text or not text.strip():
        raise ContentClassificationError("Cannot classify empty text.")

    truncated_text = text[:_MAX_INPUT_CHARACTERS]

    try:
        classifier = _get_classifier()
        output = classifier(truncated_text, candidate_labels=_CONTENT_CLASSES, multi_label=False)
    except Exception as exc:
        raise ContentClassificationError(f"Document classification failed: {exc}") from exc

    labels: list[str] = output["labels"]
    scores: list[float] = output["scores"]

    if not labels:
        raise ContentClassificationError("Document classification returned no results.")

    result = ContentClassification(label=labels[0], confidence=round(scores[0], 4))

    logger.info("Classified document as %r with confidence %.4f.", result.label, result.confidence)
    return result