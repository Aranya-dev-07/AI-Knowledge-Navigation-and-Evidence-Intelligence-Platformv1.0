"""Webpage content extraction for the TrustLens AI Content Understanding Engine.

Single responsibility: given a webpage URL, download the page and
extract its readable textual content \u2014 stripping non-content noise
(scripts, styles, navigation, ads) while preserving heading structure
as plain-text markers. Performs no NLP cleaning, language detection,
or analysis; downstream stages (``text_cleaner`` onward) own that.
"""

import logging

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: float = 15.0
_DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; TrustLensAI-Bot/1.0; "
    "+https://trustlens.ai/bot)"
)
_NOISE_TAGS: frozenset[str] = frozenset(
    {"script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe", "svg"}
)
_HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TEXT_TAGS: frozenset[str] = frozenset({"p", "li", "blockquote", "td", "th"})


class WebpageParsingError(Exception):
    """Raised when a webpage cannot be downloaded or parsed."""


def fetch_html(url: str, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> str:
    """Download the raw HTML of a webpage.

    Args:
        url: The fully qualified HTTP(S) URL to download.
        timeout_seconds: Maximum time, in seconds, to wait for the
            request to complete. Defaults to 15 seconds.

    Raises:
        WebpageParsingError: If the request fails, times out, or
            returns a non-success HTTP status code.

    Returns:
        str: The raw HTML document as text.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": _DEFAULT_USER_AGENT},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebpageParsingError(f"Failed to download webpage '{url}': {exc}") from exc

    return response.text


def _iter_readable_lines(soup: BeautifulSoup) -> list[str]:
    """Walk parsed HTML and collect readable lines, marking headings.

    Args:
        soup: The parsed ``BeautifulSoup`` document, already stripped
            of noise tags.

    Returns:
        list[str]: Ordered text lines, with heading lines prefixed by
            a repeated ``#`` marker corresponding to their heading
            level (e.g. ``"## Section Title"`` for an ``<h2>``).
    """
    lines: list[str] = []
    body = soup.body or soup

    for element in body.find_all(True):
        if not isinstance(element, Tag):
            continue

        tag_name = element.name.lower()

        if tag_name in _HEADING_TAGS:
            text = element.get_text(strip=True)
            if text:
                level = int(tag_name[1])
                lines.append(f"{'#' * level} {text}")
        elif tag_name in _BLOCK_TEXT_TAGS:
            text = element.get_text(strip=True)
            if text:
                lines.append(text)

    return lines


def extract_readable_text(html: str) -> str:
    """Extract readable, heading-aware plain text from raw HTML.

    Removes non-content elements (scripts, styles, navigation, forms,
    embedded frames) and walks the remaining document, preserving
    heading levels as ``#``-prefixed markers and joining block-level
    text elements (paragraphs, list items, table cells, quotes) as
    separate lines.

    Args:
        html: The raw HTML document to extract text from.

    Returns:
        str: The extracted readable text, with lines separated by
            newlines. Returns an empty string if no readable content
            is found.
    """
    soup = BeautifulSoup(html, "html.parser")

    for noise_tag in soup.find_all(list(_NOISE_TAGS)):
        noise_tag.decompose()

    lines = _iter_readable_lines(soup)
    return "\n".join(lines)


def parse_webpage(url: str, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> str:
    """Download a webpage and extract its readable text content.

    This is the single public entry point for this module, called by
    ``content_loader.load_content`` for inputs detected as webpage URLs.

    Args:
        url: The fully qualified HTTP(S) URL of the webpage to parse.
        timeout_seconds: Maximum time, in seconds, to wait for the
            download to complete. Defaults to 15 seconds.

    Raises:
        WebpageParsingError: If the webpage cannot be downloaded, or
            no readable text content could be extracted from it.

    Returns:
        str: The extracted raw readable text of the webpage, with
            heading structure preserved via ``#``-prefixed markers.
    """
    logger.info("Parsing webpage: %s", url)
    html = fetch_html(url, timeout_seconds=timeout_seconds)
    text = extract_readable_text(html)

    if not text.strip():
        raise WebpageParsingError(f"No readable text content found at '{url}'.")

    logger.info("Extracted %d characters of readable text from '%s'.", len(text), url)
    return text