"""Generic, reusable helper utilities for TrustLens AI backend.

Contains small, dependency-free (stdlib-only) utility functions that
are broadly useful across the codebase \u2014 string manipulation, time
handling, collection chunking, and light data-masking for logging.
Deliberately excludes anything related to authentication, database
access, or domain/business logic; those belong in ``core.security``,
``database``, and ``services`` respectively.
"""

import re
import uuid
from collections.abc import Generator, Iterable, Sequence
from datetime import UTC, datetime
from typing import TypeVar

T = TypeVar("T")

_SLUGIFY_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_CAMEL_CASE_PATTERN = re.compile(r"_([a-z0-9])")
_EMAIL_PATTERN = re.compile(r"^(.)(.*)(@.+)$")


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        datetime: The current moment, with ``tzinfo`` set to UTC.
    """
    return datetime.now(UTC)


def generate_uuid() -> str:
    """Generate a new random UUID4 string.

    Returns:
        str: A UUID4 value formatted as a hyphenated hexadecimal string.
    """
    return str(uuid.uuid4())


def slugify(text: str) -> str:
    """Convert arbitrary text into a URL-safe, lowercase slug.

    Non-alphanumeric characters are collapsed into single hyphens and
    leading/trailing hyphens are stripped.

    Args:
        text: The input string to slugify.

    Returns:
        str: The slugified string. Returns an empty string if ``text``
            contains no alphanumeric characters.
    """
    normalized = _SLUGIFY_NON_ALNUM_PATTERN.sub("-", text.strip().lower())
    return normalized.strip("-")


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate a string to a maximum length, appending a suffix if cut.

    Args:
        text: The input string to truncate.
        max_length: The maximum allowed length of the returned string,
            including the suffix.
        suffix: The marker appended when truncation occurs. Defaults
            to ``"..."``.

    Raises:
        ValueError: If ``max_length`` is shorter than ``suffix``.

    Returns:
        str: The original string if it already fits within
            ``max_length``, otherwise a truncated version ending in
            ``suffix``.
    """
    if max_length < len(suffix):
        raise ValueError("max_length must be at least as long as the suffix.")
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def mask_email(email: str) -> str:
    """Partially mask an email address for safe logging or display.

    Example:
        ``"jane.doe@example.com"`` becomes ``"j***@example.com"``.

    Args:
        email: The email address to mask.

    Returns:
        str: The masked email address. If ``email`` does not match a
            basic ``local@domain`` shape, it is returned unchanged.
    """
    match = _EMAIL_PATTERN.match(email)
    if not match:
        return email
    first_char, _rest, domain_part = match.groups()
    return f"{first_char}***{domain_part}"


def chunk_list(items: Sequence[T], chunk_size: int) -> Generator[Sequence[T], None, None]:
    """Yield successive fixed-size chunks from a sequence.

    Args:
        items: The sequence to split into chunks.
        chunk_size: The maximum number of items per chunk. Must be
            greater than zero.

    Raises:
        ValueError: If ``chunk_size`` is not greater than zero.

    Yields:
        Sequence[T]: Successive slices of ``items``, each of length
            ``chunk_size`` except possibly the last.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def dedupe_preserve_order(items: Iterable[T]) -> list[T]:
    """Remove duplicate items from an iterable while preserving order.

    Args:
        items: An iterable of hashable items, possibly containing
            duplicates.

    Returns:
        list[T]: A new list containing only the first occurrence of
            each distinct item, in original order.
    """
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def snake_to_camel(snake_str: str) -> str:
    """Convert a snake_case string to lowerCamelCase.

    Args:
        snake_str: The snake_case input string (e.g. ``"full_name"``).

    Returns:
        str: The camelCase equivalent (e.g. ``"fullName"``).
    """
    return _CAMEL_CASE_PATTERN.sub(lambda m: m.group(1).upper(), snake_str)


def safe_int(value: object, default: int | None = None) -> int | None:
    """Attempt to convert a value to ``int``, returning a default on failure.

    Args:
        value: The value to convert. Commonly a string, float, or
            already an int.
        default: The value to return if conversion fails. Defaults to
            ``None``.

    Returns:
        int | None: The converted integer, or ``default`` if ``value``
            could not be converted.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default