"""Image OCR extraction for the TrustLens AI Content Understanding Engine.

Single responsibility: given a path to an image file, load it,
apply lightweight quality-improving preprocessing where it helps OCR
accuracy, run text recognition, and return the extracted raw text.
Performs no NLP cleaning or analysis; downstream stages own that.
"""

import logging

import cv2
import easyocr
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

_OCR_LANGUAGES: list[str] = ["en"]
_MIN_UPSCALE_WIDTH_PX: int = 800
_UPSCALE_INTERPOLATION = cv2.INTER_CUBIC

_reader: easyocr.Reader | None = None


class ImageOCRError(Exception):
    """Raised when an image cannot be read or no text can be extracted from it."""


def _get_reader() -> easyocr.Reader:
    """Return a lazily initialized, process-wide EasyOCR reader.

    EasyOCR's model loading is expensive, so the reader is
    constructed once per process and reused across calls rather than
    being recreated on every invocation.

    Returns:
        easyocr.Reader: The shared OCR reader instance.
    """
    global _reader
    if _reader is None:
        logger.info("Initializing EasyOCR reader for languages: %s", _OCR_LANGUAGES)
        _reader = easyocr.Reader(_OCR_LANGUAGES, gpu=False)
    return _reader


def _load_image(path: str) -> np.ndarray:
    """Load an image file and convert it to an OpenCV-compatible array.

    Args:
        path: Filesystem path to the image file.

    Raises:
        ImageOCRError: If the file does not exist, is not a valid
            image, or cannot be read.

    Returns:
        np.ndarray: The image as a BGR NumPy array, as expected by
            OpenCV.
    """
    try:
        with Image.open(path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            array = np.array(rgb_image)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ImageOCRError(f"Failed to read image '{path}': {exc}") from exc

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Improve image quality for OCR when beneficial.

    Applies a lightweight, generally-safe preprocessing pipeline:
    grayscale conversion, denoising, adaptive thresholding for
    contrast, and upscaling of unusually small images (which tend to
    produce poor OCR results at low resolution).

    Args:
        image: The source image as a BGR NumPy array.

    Returns:
        np.ndarray: The preprocessed, OCR-ready grayscale image.
    """
    height, width = image.shape[:2]
    if width < _MIN_UPSCALE_WIDTH_PX:
        scale_factor = _MIN_UPSCALE_WIDTH_PX / width
        new_size = (int(width * scale_factor), int(height * scale_factor))
        image = cv2.resize(image, new_size, interpolation=_UPSCALE_INTERPOLATION)
        logger.debug("Upscaled small image from width=%d to width=%d.", width, new_size[0])

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(grayscale, h=10)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return thresholded


def extract_text_from_image(path: str) -> str:
    """Extract text from an image via OCR.

    This is the single public entry point for this module, called by
    ``content_loader.load_content`` for inputs detected as image files.

    Args:
        path: Filesystem path to the image file to process.

    Raises:
        ImageOCRError: If the image cannot be read, or OCR yields no
            recognizable text.

    Returns:
        str: The extracted text, with each detected text region on
            its own line, in the order EasyOCR detected them.
    """
    logger.info("Running OCR on image: %s", path)

    raw_image = _load_image(path)
    preprocessed = preprocess_image(raw_image)

    try:
        reader = _get_reader()
        detected_lines: list[str] = reader.readtext(preprocessed, detail=0)
    except Exception as exc:
        raise ImageOCRError(f"OCR failed for image '{path}': {exc}") from exc

    text = "\n".join(line.strip() for line in detected_lines if line.strip())

    if not text:
        raise ImageOCRError(f"No text could be detected in image '{path}'.")

    logger.info("Extracted %d characters of text from image '%s'.", len(text), path)
    return text