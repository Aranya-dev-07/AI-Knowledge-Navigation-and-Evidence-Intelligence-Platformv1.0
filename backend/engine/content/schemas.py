"""Shared data schemas for the TrustLens AI Content Understanding Engine.

Defines every Pydantic model passed between pipeline stages \u2014 from raw
content loading through to the final structured JSON output. This
module is the content engine's single data-contract spine: every
other module in ``engines.content`` imports its input/output types
from here rather than defining ad hoc structures, keeping every stage
interoperable with ``pipeline.py`` and with each other.

Pipeline data flow (matching the architecture diagram):

    RawContent -> CleanContent -> ContentMetadata
                                -> LanguageDetectionResult
                                -> TopicClassificationResult
                                -> list[Keyword]
                                -> list[NamedEntity]
                                -> list[Claim]
                                -> ContentClassification
                                -> FinalContent (aggregate JSON output)
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class ContentType(StrEnum):
    """Enumerates the categories of input the Content Engine can process."""

    TEXT = "text"
    WEBPAGE = "webpage"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RawContent(BaseModel):
    """Uniform envelope for raw, unprocessed content returned by ``content_loader``.

    Attributes:
        content_type: The detected (or explicitly specified) type of
            the original input.
        source: The original input reference \u2014 a URL, filesystem
            path, or the raw text itself for plain-text input.
        raw_text: The extracted raw text content, prior to any
            cleaning or normalization.
    """

    content_type: ContentType = Field(..., description="Detected content type of the input.")
    source: str = Field(..., description="Original input reference (URL, path, or raw text).")
    raw_text: str = Field(..., description="Extracted raw text, prior to cleaning.")


class CleanContent(BaseModel):
    """Normalized, cleaned text produced by ``text_cleaner``.

    Attributes:
        cleaned_text: The whitespace-collapsed, Unicode-normalized,
            control-character-stripped text.
        character_count: The length of ``cleaned_text``, computed
            automatically.
    """

    cleaned_text: str = Field(..., description="Cleaned, normalized text content.")

    @computed_field(description="Number of characters in the cleaned text.")  # type: ignore[misc]
    @property
    def character_count(self) -> int:
        """Compute the character count of the cleaned text.

        Returns:
            int: The length of ``cleaned_text``.
        """
        return len(self.cleaned_text)


class ContentMetadata(BaseModel):
    """Structured document metadata produced by ``metadata_extractor``.

    Attributes:
        title: The document's title, if available.
        author: The document's author, if available.
        creation_date: The document's creation or publication date,
            if available.
        source: The original content source reference.
        file_size_bytes: The size of the underlying file in bytes, if
            applicable (not applicable to webpages).
        page_count: The number of pages, if applicable (PDFs only).
    """

    title: str | None = Field(default=None, description="Document title, if available.")
    author: str | None = Field(default=None, description="Document author, if available.")
    creation_date: datetime | None = Field(
        default=None, description="Document creation/publication date, if available."
    )
    source: str = Field(..., description="Original content source reference.")
    file_size_bytes: int | None = Field(
        default=None, description="File size in bytes, if applicable."
    )
    page_count: int | None = Field(
        default=None, description="Number of pages, if applicable (PDFs only)."
    )


class LanguageDetectionResult(BaseModel):
    """Detected language produced by ``language_detector``.

    Attributes:
        language: The detected ISO 639-1 language code (e.g. ``"en"``).
        confidence: The detector's confidence score, between 0.0 and 1.0.
    """

    language: str = Field(..., description="Detected ISO 639-1 language code.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score.")


class TopicScore(BaseModel):
    """A single scored topic candidate.

    Attributes:
        topic: The topic label.
        confidence: The classifier's confidence score for this topic,
            between 0.0 and 1.0.
    """

    topic: str = Field(..., description="Topic label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score.")


class TopicClassificationResult(BaseModel):
    """Topic classification output produced by ``topic_classifier``.

    Attributes:
        primary_topic: The single highest-confidence topic label.
        primary_confidence: The confidence score of ``primary_topic``.
        secondary_topics: Additional qualifying topics beyond the
            primary one, each with its own confidence score.
    """

    primary_topic: str = Field(..., description="Highest-confidence topic label.")
    primary_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the primary topic."
    )
    secondary_topics: list[TopicScore] = Field(
        default_factory=list, description="Additional qualifying topics."
    )


class Keyword(BaseModel):
    """A single ranked keyword produced by ``keyword_extractor``.

    Attributes:
        text: The keyword or keyphrase text.
        score: The keyword's relevance score, normalized between 0.0
            and 1.0 relative to the most frequent candidate in the
            same document.
    """

    text: str = Field(..., description="Keyword or keyphrase text.")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance score.")


class NamedEntity(BaseModel):
    """A single structured named entity produced by ``ner``.

    Attributes:
        text: The entity's surface text as it appears in the document.
        entity_type: The entity category (one of ``PERSON``,
            ``ORGANIZATION``, ``LOCATION``, ``DATE``, ``EVENT``,
            ``PRODUCT``, ``MONEY``, ``QUANTITY``).
        count: The number of times this entity (by text and category)
            occurs in the document.
    """

    text: str = Field(..., description="Entity surface text.")
    entity_type: str = Field(..., description="Entity category.")
    count: int = Field(..., ge=1, description="Occurrence count within the document.")


class Claim(BaseModel):
    """A single candidate factual claim produced by ``claim_extractor``.

    Attributes:
        text: The sentence identified as a candidate factual claim.
        confidence: The classifier's confidence that this sentence is
            a factual statement rather than an opinion, between 0.0
            and 1.0. This is NOT a truthfulness score \u2014 claim
            verification is out of scope for this module.
    """

    text: str = Field(..., description="Candidate factual claim sentence.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence that this is a factual statement (not truth)."
    )


class ContentClassification(BaseModel):
    """Document-type classification produced by ``content_classifier``.

    Attributes:
        label: The classified document category (one of ``News``,
            ``Research Paper``, ``Government Report``, ``Blog``,
            ``Social Media``, ``Documentation``, ``Legal``,
            ``Medical``, ``Educational``, ``Other``).
        confidence: The classifier's confidence score, between 0.0
            and 1.0.
    """

    label: str = Field(..., description="Classified document category.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score.")


class FinalContent(BaseModel):
    """The complete, structured output of the Content Understanding Engine.

    Aggregates the results of every pipeline stage into a single
    structured JSON-serializable object \u2014 the terminal artifact
    produced by ``pipeline.py`` for a single piece of processed content.

    Attributes:
        source: The original content source reference.
        content_type: The detected content type of the original input.
        content: The cleaned, normalized text content.
        metadata: Structured document metadata.
        language: Detected language and confidence.
        topics: Primary and secondary topic classification.
        keywords: Ranked extracted keywords.
        entities: Structured named entities.
        claims: Candidate factual claims (unverified).
        classification: Overall document-type classification.
        processed_at: UTC timestamp marking when processing completed.
    """

    source: str = Field(..., description="Original content source reference.")
    content_type: ContentType = Field(..., description="Detected content type of the input.")
    content: CleanContent = Field(..., description="Cleaned, normalized text content.")
    metadata: ContentMetadata = Field(..., description="Structured document metadata.")
    language: LanguageDetectionResult = Field(..., description="Detected language and confidence.")
    topics: TopicClassificationResult = Field(
        ..., description="Primary and secondary topic classification."
    )
    keywords: list[Keyword] = Field(default_factory=list, description="Ranked extracted keywords.")
    entities: list[NamedEntity] = Field(
        default_factory=list, description="Structured named entities."
    )
    claims: list[Claim] = Field(
        default_factory=list, description="Candidate factual claims (unverified)."
    )
    classification: ContentClassification = Field(
        ..., description="Overall document-type classification."
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp marking when processing completed.",
    )