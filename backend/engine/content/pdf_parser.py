"""PDF content extraction for the TrustLens AI Content Understanding Engine.

Single responsibility: given a path to a PDF file, extract its raw
text content page by page (in original page order), transparently
handling password-protected/encrypted PDFs where possible. Performs
no NLP cleaning or analysis; downstream stages own that.
"""

import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_PAGE_MARKER_TEMPLATE: str = "[Page {page_number}]"


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened, decrypted, or read."""


def _open_pdf(path: str) -> fitz.Document:
    """Open a PDF document, transparently handling encryption.

    Attempts to decrypt password-protected PDFs using an empty
    password first, as many "encrypted" PDFs in the wild are merely
    permission-restricted (no owner password required to read text).

    Args:
        path: Filesystem path to the PDF file.

    Raises:
        PDFParsingError: If the file cannot be opened, or is encrypted
            with a non-empty password that cannot be bypassed.

    Returns:
        fitz.Document: The opened, readable PDF document.
    """
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PDFParsingError(f"Failed to open PDF '{path}': {exc}") from exc

    if document.is_encrypted:
        decrypted = document.authenticate("")
        if not decrypted:
            document.close()
            raise PDFParsingError(
                f"PDF '{path}' is encrypted with a password and could not be decrypted."
            )
        logger.info("Decrypted password-protected PDF '%s' using an empty password.", path)

    return document


def parse_pdf(path: str) -> str:
    """Extract text from every page of a PDF, preserving page order.

    This is the single public entry point for this module, called by
    ``content_loader.load_content`` for inputs detected as PDF files.

    Args:
        path: Filesystem path to the PDF file to parse.

    Raises:
        PDFParsingError: If the PDF cannot be opened, is encrypted and
            cannot be decrypted, or contains no extractable text.

    Returns:
        str: The concatenated extracted text of all pages, in
            original page order, with each page's text preceded by a
            ``[Page N]`` marker to preserve page-boundary information
            for downstream stages (e.g. metadata extraction).
    """
    logger.info("Parsing PDF: %s", path)
    document = _open_pdf(path)

    try:
        page_texts: list[str] = []
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()
            if page_text:
                marker = _PAGE_MARKER_TEMPLATE.format(page_number=page_number)
                page_texts.append(f"{marker}\n{page_text}")
    except Exception as exc:
        raise PDFParsingError(f"Failed to read pages from PDF '{path}': {exc}") from exc
    finally:
        document.close()

    if not page_texts:
        raise PDFParsingError(f"No extractable text found in PDF '{path}'.")

    full_text = "\n\n".join(page_texts)
    logger.info(
        "Extracted %d characters across %d page(s) from PDF '%s'.",
        len(full_text),
        len(page_texts),
        path,
    )
    return full_text