"""Audio/video transcription for the TrustLens AI Content Understanding Engine.

Single responsibility: given a path to an audio or video file, load
it, extract and transcribe its speech using OpenAI Whisper, and
return the resulting transcript text. Handles both audio and video
inputs identically, since Whisper extracts the audio track from video
containers internally (via ffmpeg). Performs no NLP cleaning or
analysis; downstream stages own that.
"""

import logging
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)

_MODEL_SIZE: str = "base"

_model: whisper.Whisper | None = None


class VideoTranscriptionError(Exception):
    """Raised when media cannot be loaded or transcribed."""


def _get_model() -> whisper.Whisper:
    """Return a lazily initialized, process-wide Whisper model.

    Loading Whisper's model weights is expensive, so the model is
    constructed once per process and reused across calls rather than
    being reloaded on every invocation.

    Returns:
        whisper.Whisper: The shared Whisper model instance.
    """
    global _model
    if _model is None:
        logger.info("Loading Whisper model: %s", _MODEL_SIZE)
        _model = whisper.load_model(_MODEL_SIZE)
    return _model


def transcribe(path: str) -> str:
    """Transcribe speech from an audio or video file into text.

    This is the single public entry point for this module, called by
    ``content_loader.load_content`` for inputs detected as audio or
    video files.

    Args:
        path: Filesystem path to the audio or video file to transcribe.

    Raises:
        VideoTranscriptionError: If the file does not exist, cannot be
            read/decoded (e.g. missing ffmpeg, corrupt media), or
            contains no detectable speech.

    Returns:
        str: The full transcript text, with segments concatenated in
            chronological order.
    """
    media_path = Path(path)
    if not media_path.is_file():
        raise VideoTranscriptionError(f"Media file not found: '{path}'.")

    logger.info("Transcribing media: %s", path)

    try:
        model = _get_model()
        result = model.transcribe(str(media_path))
    except Exception as exc:
        raise VideoTranscriptionError(f"Failed to transcribe media '{path}': {exc}") from exc

    transcript = str(result.get("text", "")).strip()

    if not transcript:
        raise VideoTranscriptionError(f"No speech detected in media '{path}'.")

    logger.info("Generated %d-character transcript from '%s'.", len(transcript), path)
    return transcript