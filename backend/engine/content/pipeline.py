"""Orchestration pipeline for the TrustLens AI Content Understanding Engine.

The sole module responsible for wiring together every independent
Content Engine stage \u2014 loading, cleaning, metadata extraction,
language detection, topic classification, keyword extraction, named
entity recognition, claim extraction, and document classification \u2014
into one deterministic execution order, producing a single structured
``FinalContent`` JSON-serializable result.

No individual stage module calls another (aside from
``content_loader``'s own internal routing to its four specialized
parsers, which is that module's own stated responsibility). All
cross-stage sequencing, data hand-off, error handling, and logging
live here and only here, keeping every other module independently
testable and reusable outside this pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from engines.content import (
    claim_extractor,
    content_classifier,
    content_loader,
    keyword_extractor,
    language_detector,
    metadata_extractor,
    ner,
    text_cleaner,
    topic_classifier,
)
from engines.content.schemas import (
    Claim,
    CleanContent,
    ContentClassification,
    ContentMetadata,
    ContentType,
    FinalContent,
    Keyword,
    LanguageDetectionResult,
    NamedEntity,
    RawContent,
    TopicClassificationResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ContentPipelineError(Exception):
    """Raised when a pipeline stage fails, wrapping the originating stage and error.

    Attributes:
        stage: The name of the pipeline stage that failed.
    """

    def __init__(self, stage: str, message: str) -> None:
        """Initialize the error with its failing stage and message.

        Args:
            stage: The name of the pipeline stage that failed.
            message: A human-readable description of the failure.
        """
        self.stage = stage
        super().__init__(f"Content pipeline failed at stage '{stage}': {message}")


@dataclass(slots=True)
class _StageTimings:
    """Accumulates per-stage execution durations for observability.

    Attributes:
        durations_seconds: Mapping of stage name to elapsed seconds.
    """

    durations_seconds: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, elapsed_seconds: float) -> None:
        """Record the elapsed duration of a completed stage.

        Args:
            stage: The name of the completed stage.
            elapsed_seconds: The time the stage took to execute, in
                seconds.
        """
        self.durations_seconds[stage] = round(elapsed_seconds, 4)


def _run_stage(
    stage_name: str, timings: _StageTimings, func: Callable[..., T], *args: Any
) -> T:
    """Execute a single pipeline stage with centralized timing, logging, and error handling.

    Args:
        stage_name: A human-readable name identifying this stage, used
            in logs and in ``ContentPipelineError`` if the stage fails.
        timings: The shared timing accumulator for this pipeline run.
        func: The stage function to invoke.
        *args: Positional arguments to pass to ``func``.

    Raises:
        ContentPipelineError: If ``func`` raises any exception.

    Returns:
        T: The return value of ``func``.
    """
    logger.info("Running pipeline stage: %s", stage_name)
    started_at = time.perf_counter()

    try:
        result = func(*args)
    except Exception as exc:
        logger.exception("Pipeline stage '%s' failed.", stage_name)
        raise ContentPipelineError(stage_name, str(exc)) from exc

    elapsed_seconds = time.perf_counter() - started_at
    timings.record(stage_name, elapsed_seconds)
    logger.info("Completed pipeline stage '%s' in %.3fs.", stage_name, elapsed_seconds)

    return result


def process_content(source: str, content_type: ContentType | None = None) -> FinalContent:
    """Run the full Content Understanding Engine pipeline on a single input.

    Executes, in strict order: content loading (which internally
    routes to the appropriate specialized parser), text cleaning,
    metadata extraction, language detection, topic classification,
    keyword extraction, named entity recognition, claim extraction,
    and document classification \u2014 then assembles every stage's output
    into a single ``FinalContent`` structured result.

    Args:
        source: The raw input \u2014 plain text, a webpage URL, or a local
            filesystem path to a PDF, image, audio, or video file.
        content_type: An explicit content type to bypass auto-detection
            in the content loading stage. If ``None`` (the default),
            the type is inferred from ``source``.

    Raises:
        ContentPipelineError: If any pipeline stage fails. The
            exception's ``stage`` attribute identifies which stage was
            responsible.

    Returns:
        FinalContent: The complete, structured output of the pipeline,
            aggregating every stage's results.
    """
    timings = _StageTimings()
    pipeline_started_at = time.perf_counter()

    logger.info("Starting content pipeline for source=%s", source[:120])

    raw_content: RawContent = _run_stage(
        "content_loading", timings, content_loader.load_content, source, content_type
    )

    cleaned_text: str = _run_stage(
        "text_cleaning", timings, text_cleaner.clean_text, raw_content.raw_text
    )
    clean_content = CleanContent(cleaned_text=cleaned_text)

    metadata: ContentMetadata = _run_stage(
        "metadata_extraction",
        timings,
        metadata_extractor.extract_metadata,
        raw_content.source,
        raw_content.content_type,
        raw_content.raw_text,
    )

    language: LanguageDetectionResult = _run_stage(
        "language_detection", timings, language_detector.detect_language, cleaned_text
    )

    topics: TopicClassificationResult = _run_stage(
        "topic_classification", timings, topic_classifier.classify_topics, cleaned_text
    )

    keywords: list[Keyword] = _run_stage(
        "keyword_extraction", timings, keyword_extractor.extract_keywords, cleaned_text
    )

    entities: list[NamedEntity] = _run_stage(
        "named_entity_recognition", timings, ner.extract_entities, cleaned_text
    )

    claims: list[Claim] = _run_stage(
        "claim_extraction", timings, claim_extractor.extract_claims, cleaned_text
    )

    classification: ContentClassification = _run_stage(
        "content_classification", timings, content_classifier.classify_content, cleaned_text
    )

    final_content = FinalContent(
        source=raw_content.source,
        content_type=raw_content.content_type,
        content=clean_content,
        metadata=metadata,
        language=language,
        topics=topics,
        keywords=keywords,
        entities=entities,
        claims=claims,
        classification=classification,
    )

    total_elapsed_seconds = time.perf_counter() - pipeline_started_at
    logger.info(
        "Content pipeline completed in %.3fs for source=%s. Stage timings: %s",
        total_elapsed_seconds,
        source[:120],
        timings.durations_seconds,
    )

    return final_content