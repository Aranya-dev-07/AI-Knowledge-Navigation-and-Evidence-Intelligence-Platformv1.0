"""Content loading and routing for the TrustLens AI Content Understanding Engine.

Single responsibility: accept raw user input (plain text, a webpage
URL, or a local file path to a PDF/image/audio/video file), validate
it, detect its content type, and dispatch to the appropriate
specialized parser module to obtain raw, unprocessed extracted
content. Performs no cleaning, analysis, or downstream pipeline
orchestration \u2014 ``pipeline.py`` is solely responsible for what happens
to the ``RawContent`` this module returns.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

from engines.content import image_ocr, pdf_parser, video_transcriber, webpage_parser
from engines.content.schemas import ContentType, RawContent

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"})
_PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"})
_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


class ContentLoadError(Exception):
    """Raised when input validation fails or content cannot be loaded."""


class UnsupportedContentTypeError(ContentLoadError):
    """Raised when the input's content type cannot be determined or is unsupported."""


def detect_content_type(source: str) -> ContentType:
    """Infer the ``ContentType`` of a given input source.

    Detection order: a well-formed HTTP(S) URL is treated as a
    webpage; otherwise, if the source resolves to an existing local
    file path, its extension determines the type (PDF, image, audio,
    or video); otherwise, the source is treated as plain text.

    Args:
        source: The raw input \u2014 a block of text, a URL, or a
            filesystem path (as a string).

    Raises:
        UnsupportedContentTypeError: If ``source`` resolves to a local
            file with an extension not in any supported category.

    Returns:
        ContentType: The detected content type.
    """
    stripped = source.strip()

    parsed = urlparse(stripped)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return ContentType.WEBPAGE

    path = Path(stripped)
    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        if suffix in _PDF_EXTENSIONS:
            return ContentType.PDF
        if suffix in _IMAGE_EXTENSIONS:
            return ContentType.IMAGE
        if suffix in _AUDIO_EXTENSIONS:
            return ContentType.AUDIO
        if suffix in _VIDEO_EXTENSIONS:
            return ContentType.VIDEO
        raise UnsupportedContentTypeError(
            f"File '{path.name}' has an unsupported extension: '{suffix}'."
        )

    return ContentType.TEXT


def validate_input(source: str) -> str:
    """Validate raw input prior to loading.

    Args:
        source: The raw input string supplied by the caller.

    Raises:
        ContentLoadError: If ``source`` is empty, whitespace-only, or
            not a string.

    Returns:
        str: The stripped, validated input string.
    """
    if not isinstance(source, str):
        raise ContentLoadError(f"Input must be a string, got {type(source).__name__}.")

    stripped = source.strip()
    if not stripped:
        raise ContentLoadError("Input must not be empty or whitespace-only.")

    return stripped


def load_content(source: str, content_type: ContentType | None = None) -> RawContent:
    """Load and route input to the correct parser, returning raw content.

    This is the single public entry point for this module and the
    only function ``pipeline.py`` should call. It validates the
    input, detects (or accepts an explicitly provided) content type,
    dispatches to the matching specialized parser, and wraps the
    result in a uniform ``RawContent`` envelope.

    Args:
        source: The raw input \u2014 plain text, a webpage URL, or a
            local filesystem path to a PDF, image, audio, or video file.
        content_type: An explicit content type to bypass auto-detection.
            If ``None`` (the default), the type is inferred via
            ``detect_content_type``.

    Raises:
        ContentLoadError: If input validation fails or the underlying
            parser is unable to extract content.
        UnsupportedContentTypeError: If the content type cannot be
            determined or is not supported.

    Returns:
        RawContent: A uniform envelope containing the extracted raw
            text/content, the detected content type, and the original
            source reference.
    """
    validated_source = validate_input(source)
    detected_type = content_type or detect_content_type(validated_source)

    logger.info("Loading content: type=%s source=%s", detected_type.value, validated_source[:120])

    try:
        match detected_type:
            case ContentType.TEXT:
                raw_text = validated_source
            case ContentType.WEBPAGE:
                raw_text = webpage_parser.parse_webpage(validated_source)
            case ContentType.PDF:
                raw_text = pdf_parser.parse_pdf(validated_source)
            case ContentType.IMAGE:
                raw_text = image_ocr.extract_text_from_image(validated_source)
            case ContentType.AUDIO | ContentType.VIDEO:
                raw_text = video_transcriber.transcribe(validated_source)
            case _:
                raise UnsupportedContentTypeError(
                    f"No parser available for content type: {detected_type}."
                )
    except UnsupportedContentTypeError:
        raise
    except Exception as exc:
        logger.exception("Failed to load content from source=%s", validated_source[:120])
        raise ContentLoadError(
            f"Failed to load content of type '{detected_type.value}': {exc}"
        ) from exc

    return RawContent(
        content_type=detected_type,
        source=validated_source,
        raw_text=raw_text,
    )