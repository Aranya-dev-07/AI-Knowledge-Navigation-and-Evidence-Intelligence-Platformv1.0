"""Text normalization for the TrustLens AI Content Understanding Engine.

Single responsibility: take raw extracted text (from any source \u2014
webpage, PDF, OCR, or transcript) and normalize it into clean, uniform
plain text: collapsing redundant whitespace, normalizing Unicode
representations, and stripping non-printable/control characters.
Deliberately performs no NLP (no tokenization, language detection, or
semantic analysis) \u2014 that is the responsibility of later pipeline
stages. Sentence and paragraph structure (line breaks) is preserved.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Collapses 2+ horizontal whitespace characters (spaces/tabs) into one space.
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")

# Collapses 3+ consecutive newlines (with optional surrounding whitespace)
# down to exactly two, preserving paragraph breaks without excessive gaps.
_MULTI_BLANK_LINE_PATTERN = re.compile(r"\n\s*\n\s*\n+")

# Trailing horizontal whitespace at the end of a line.
_TRAILING_LINE_WHITESPACE_PATTERN = re.compile(r"[ \t]+(?=\n)")

# Unicode categories considered non-printable "unwanted" characters:
# Cc (control), Cf (format, e.g. zero-width joiners), Co (private use),
# Cs (surrogate). Newline and tab are handled separately and preserved.
_UNWANTED_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs"})
_PRESERVED_WHITESPACE_CHARS = frozenset({"\n", "\t"})


class TextCleaningError(Exception):
    """Raised when input is invalid or cleaning yields no usable text."""


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to a single canonical representation.

    Applies NFKC normalization, which combines canonical decomposition
    with compatibility composition \u2014 e.g. collapsing visually
    equivalent character sequences (full-width characters, ligatures,
    combining accents) into their standard composed form.

    Args:
        text: The input text to normalize.

    Returns:
        str: The NFKC-normalized text.
    """
    return unicodedata.normalize("NFKC", text)


def remove_unwanted_characters(text: str) -> str:
    """Strip non-printable control/format characters from text.

    Removes Unicode control, format, private-use, and surrogate
    characters (e.g. zero-width spaces, byte-order marks, embedded
    control codes) while explicitly preserving newlines and tabs,
    which carry structural meaning.

    Args:
        text: The input text to strip.

    Returns:
        str: The text with unwanted non-printable characters removed.
    """
    return "".join(
        char
        for char in text
        if char in _PRESERVED_WHITESPACE_CHARS
        or unicodedata.category(char) not in _UNWANTED_UNICODE_CATEGORIES
    )


def collapse_whitespace(text: str) -> str:
    """Collapse redundant whitespace while preserving sentence/paragraph structure.

    Repeated spaces or tabs are collapsed to a single space, trailing
    whitespace at line ends is removed, and runs of three or more
    newlines are collapsed to exactly two (a single blank line),
    preserving paragraph breaks without excessive vertical gaps.

    Args:
        text: The input text to collapse whitespace in.

    Returns:
        str: The text with redundant whitespace collapsed.
    """
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    text = _TRAILING_LINE_WHITESPACE_PATTERN.sub("", text)
    text = _MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
    return text


def clean_text(raw_text: str) -> str:
    """Normalize raw extracted text into clean, uniform plain text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` immediately after content loading. Applies, in
    order: Unicode normalization, unwanted-character stripping, and
    whitespace collapsing. Sentence and paragraph line breaks are
    preserved throughout.

    Args:
        raw_text: The raw text to clean, as extracted by
            ``content_loader`` (or one of its underlying parsers).

    Raises:
        TextCleaningError: If ``raw_text`` is not a string, or cleaning
            leaves no usable text content.

    Returns:
        str: The cleaned, normalized text, stripped of leading and
            trailing whitespace.
    """
    if not isinstance(raw_text, str):
        raise TextCleaningError(f"Input must be a string, got {type(raw_text).__name__}.")

    normalized = normalize_unicode(raw_text)
    stripped_of_unwanted = remove_unwanted_characters(normalized)
    collapsed = collapse_whitespace(stripped_of_unwanted)
    cleaned = collapsed.strip()

    if not cleaned:
        raise TextCleaningError("Cleaning left no usable text content.")

    logger.info("Cleaned text: %d characters -> %d characters.", len(raw_text), len(cleaned))
    return cleaned