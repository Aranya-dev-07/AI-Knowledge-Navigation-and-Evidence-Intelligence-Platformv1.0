"""Metadata extraction for the TrustLens AI Content Understanding Engine.

Single responsibility: given a content source and its detected type,
extract structured document metadata \u2014 title, author, creation date,
source reference, file size, and page count where applicable. Each
content type has different metadata availability (e.g. PDFs carry
embedded document metadata; plain text carries none), so extraction
is best-effort per field: a missing individual field never fails the
whole extraction. Performs no text analysis \u2014 that is the
responsibility of later pipeline stages.
"""

import logging
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from PIL import ExifTags, Image, UnidentifiedImageError

from engines.content.schemas import ContentMetadata, ContentType

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: float = 10.0
_DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; TrustLensAI-Bot/1.0; +https://trustlens.ai/bot)"
)
_HEADING_LINE_PREFIX = "#"
_PDF_DATE_PREFIX = "D:"


class MetadataExtractionError(Exception):
    """Raised when metadata extraction receives fundamentally invalid input."""


def _derive_title_from_text(raw_text: str) -> str | None:
    """Best-effort title derivation from the first heading or line of text.

    Args:
        raw_text: The extracted (or cleaned) text content to scan.

    Returns:
        str | None: The first heading line (with ``#`` markers
            stripped) if present, otherwise the first non-empty line
            of text truncated to a reasonable title length, or
            ``None`` if no usable text is found.
    """
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_HEADING_LINE_PREFIX):
            return stripped.lstrip("#").strip() or None
        return stripped[:200]
    return None


def _parse_pdf_date(raw_date: str | None) -> datetime | None:
    """Parse a PDF-format date string (e.g. ``D:20240115120000``) into a datetime.

    Args:
        raw_date: The raw date string as stored in PDF document
            metadata, or ``None``.

    Returns:
        datetime | None: The parsed datetime, or ``None`` if ``raw_date``
            is missing or not in a recognizable PDF date format.
    """
    if not raw_date:
        return None

    value = raw_date[len(_PDF_DATE_PREFIX) :] if raw_date.startswith(_PDF_DATE_PREFIX) else raw_date
    date_part = value[:14]

    try:
        return datetime.strptime(date_part, "%Y%m%d%H%M%S")
    except ValueError:
        logger.debug("Could not parse PDF date string: %r", raw_date)
        return None


def _extract_pdf_metadata(source: str, raw_text: str) -> ContentMetadata:
    """Extract structured metadata from a PDF file.

    Args:
        source: Filesystem path to the PDF file.
        raw_text: The previously extracted raw text, used as a title
            fallback if the PDF has no embedded title.

    Returns:
        ContentMetadata: Best-effort metadata for the PDF.
    """
    path = Path(source)
    file_size = path.stat().st_size if path.is_file() else None

    title: str | None = None
    author: str | None = None
    creation_date: datetime | None = None
    page_count: int | None = None

    try:
        document = fitz.open(source)
        try:
            pdf_meta = document.metadata or {}
            title = (pdf_meta.get("title") or "").strip() or None
            author = (pdf_meta.get("author") or "").strip() or None
            creation_date = _parse_pdf_date(pdf_meta.get("creationDate"))
            page_count = document.page_count
        finally:
            document.close()
    except Exception:
        logger.warning("Failed to read embedded PDF metadata for '%s'.", source, exc_info=True)

    return ContentMetadata(
        title=title or _derive_title_from_text(raw_text),
        author=author,
        creation_date=creation_date,
        source=source,
        file_size_bytes=file_size,
        page_count=page_count,
    )


def _extract_webpage_metadata(source: str, raw_text: str) -> ContentMetadata:
    """Extract structured metadata from a webpage.

    Performs a lightweight, independent HTTP request to read
    ``<title>`` and common meta tags. Kept self-contained (does not
    depend on ``webpage_parser``'s internals) so this module remains
    independently usable; any failure here is non-fatal and falls
    back to text-derived values.

    Args:
        source: The webpage URL.
        raw_text: The previously extracted raw text, used as a title
            fallback.

    Returns:
        ContentMetadata: Best-effort metadata for the webpage.
    """
    title: str | None = None
    author: str | None = None
    creation_date: datetime | None = None

    try:
        response = requests.get(
            source, headers={"User-Agent": _DEFAULT_USER_AGENT}, timeout=_DEFAULT_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        if soup.title and soup.title.string:
            title = soup.title.string.strip() or None

        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag and author_tag.get("content"):
            author = author_tag["content"].strip() or None

        published_tag = soup.find("meta", attrs={"property": "article:published_time"})
        if published_tag and published_tag.get("content"):
            try:
                creation_date = datetime.fromisoformat(published_tag["content"].strip())
            except ValueError:
                logger.debug("Unrecognized published-time format for '%s'.", source)
    except Exception:
        logger.warning("Failed to fetch webpage metadata for '%s'.", source, exc_info=True)

    return ContentMetadata(
        title=title or _derive_title_from_text(raw_text),
        author=author,
        creation_date=creation_date,
        source=source,
        file_size_bytes=None,
        page_count=None,
    )


def _extract_image_metadata(source: str, raw_text: str) -> ContentMetadata:
    """Extract structured metadata from an image file, including EXIF data.

    Args:
        source: Filesystem path to the image file.
        raw_text: The previously extracted OCR text, used as a title
            fallback.

    Returns:
        ContentMetadata: Best-effort metadata for the image.
    """
    path = Path(source)
    file_size = path.stat().st_size if path.is_file() else None
    creation_date: datetime | None = None

    try:
        with Image.open(source) as image:
            exif = image.getexif()
            if exif:
                tag_map = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}
                raw_date = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
                if raw_date:
                    try:
                        creation_date = datetime.strptime(str(raw_date), "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        logger.debug("Unrecognized EXIF date format for '%s'.", source)
    except (UnidentifiedImageError, OSError):
        logger.warning("Failed to read EXIF metadata for '%s'.", source, exc_info=True)

    return ContentMetadata(
        title=_derive_title_from_text(raw_text) or path.stem,
        author=None,
        creation_date=creation_date,
        source=source,
        file_size_bytes=file_size,
        page_count=None,
    )


def _extract_media_metadata(source: str, raw_text: str) -> ContentMetadata:
    """Extract structured metadata from an audio or video file.

    Limited to filesystem-derived fields (no dedicated media-tag
    library is part of this module's scope); creation date and author
    are left unset.

    Args:
        source: Filesystem path to the audio or video file.
        raw_text: The previously generated transcript, used as a title
            fallback.

    Returns:
        ContentMetadata: Best-effort metadata for the media file.
    """
    path = Path(source)
    file_size = path.stat().st_size if path.is_file() else None

    return ContentMetadata(
        title=_derive_title_from_text(raw_text) or path.stem,
        author=None,
        creation_date=None,
        source=source,
        file_size_bytes=file_size,
        page_count=None,
    )


def _extract_text_metadata(source: str, raw_text: str) -> ContentMetadata:
    """Extract structured metadata for plain-text input.

    Args:
        source: The original plain-text input.
        raw_text: The same raw text (plain-text inputs have no
            separate extraction step).

    Returns:
        ContentMetadata: Best-effort metadata for the plain-text input.
    """
    return ContentMetadata(
        title=_derive_title_from_text(raw_text),
        author=None,
        creation_date=None,
        source="inline-text",
        file_size_bytes=len(source.encode("utf-8")),
        page_count=None,
    )


def extract_metadata(source: str, content_type: ContentType, raw_text: str) -> ContentMetadata:
    """Extract structured metadata appropriate to the given content type.

    This is the single public entry point for this module, called by
    ``pipeline.py`` alongside (not instead of) text cleaning. Metadata
    extraction is best-effort: individual unavailable fields are left
    as ``None`` rather than failing the whole extraction, since
    metadata is supplementary to the primary text pipeline.

    Args:
        source: The original content source (URL, filesystem path, or
            inline text), matching ``RawContent.source``.
        content_type: The detected content type, matching
            ``RawContent.content_type``.
        raw_text: The extracted raw (or cleaned) text, used as a title
            fallback when no richer metadata is available.

    Raises:
        MetadataExtractionError: If ``content_type`` is not a
            recognized ``ContentType`` value.

    Returns:
        ContentMetadata: The best-effort structured metadata for the
            given content.
    """
    logger.info("Extracting metadata: type=%s source=%s", content_type, source[:120])

    match content_type:
        case ContentType.PDF:
            metadata = _extract_pdf_metadata(source, raw_text)
        case ContentType.WEBPAGE:
            metadata = _extract_webpage_metadata(source, raw_text)
        case ContentType.IMAGE:
            metadata = _extract_image_metadata(source, raw_text)
        case ContentType.AUDIO | ContentType.VIDEO:
            metadata = _extract_media_metadata(source, raw_text)
        case ContentType.TEXT:
            metadata = _extract_text_metadata(source, raw_text)
        case _:
            raise MetadataExtractionError(f"Unrecognized content type: {content_type!r}.")

    logger.info("Extracted metadata for source=%s: title=%r", source[:120], metadata.title)
    return metadata