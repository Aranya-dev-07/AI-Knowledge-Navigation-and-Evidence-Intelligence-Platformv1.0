"""Content Understanding Engine package for TrustLens AI backend.

Re-exports every module of the content processing pipeline \u2014 the
shared data schemas, each single-responsibility pipeline stage, and
the orchestrating ``pipeline`` module itself \u2014 so they can be imported
as ``from engines.content import content_loader, schemas`` etc., in
addition to their fully-qualified submodule paths.

Import order is deliberate: ``schemas`` (the shared data-contract leaf
that every other module depends on) and the individual, independent
pipeline-stage modules are imported first, with ``pipeline`` (the sole
module that depends on all of the others) imported last. This keeps
the dependency graph acyclic and avoids partially-initialized-module
import errors regardless of which module a caller touches first.
"""

from engines.content import (
    claim_extractor,
    content_classifier,
    content_loader,
    image_ocr,
    keyword_extractor,
    language_detector,
    metadata_extractor,
    ner,
    pdf_parser,
    schemas,
    text_cleaner,
    topic_classifier,
    video_transcriber,
    webpage_parser,
)
from engines.content import pipeline

__all__ = [
    "schemas",
    "content_loader",
    "webpage_parser",
    "pdf_parser",
    "image_ocr",
    "video_transcriber",
    "text_cleaner",
    "metadata_extractor",
    "language_detector",
    "topic_classifier",
    "keyword_extractor",
    "ner",
    "claim_extractor",
    "content_classifier",
    "pipeline",
]