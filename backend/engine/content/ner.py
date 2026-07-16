"""Named Entity Recognition for the TrustLens AI Content Understanding Engine.

Single responsibility: given cleaned text, identify and structure
named entities of interest \u2014 people, organizations, locations, dates,
events, products, monetary values, and quantities \u2014 using spaCy's
built-in NER model. Entities outside this category set (e.g.
languages, laws, percentages) are intentionally excluded to keep
output focused. Performs no claim extraction or fact verification;
downstream stages own that.
"""

import logging
from collections import Counter

import spacy
from spacy.language import Language

from engines.content.schemas import NamedEntity

logger = logging.getLogger(__name__)

_SPACY_MODEL_NAME: str = "en_core_web_sm"
_MAX_INPUT_CHARACTERS: int = 100_000

# Maps spaCy's default entity labels onto TrustLens AI's requested
# entity category set. spaCy labels not present here (e.g. NORP, FAC,
# WORK_OF_ART, LAW, LANGUAGE, TIME, PERCENT, ORDINAL, CARDINAL) are
# intentionally excluded from output.
_ENTITY_LABEL_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT",
    "MONEY": "MONEY",
    "QUANTITY": "QUANTITY",
}

_nlp: Language | None = None


class NERExtractionError(Exception):
    """Raised when named entity recognition fails."""


def _get_nlp() -> Language:
    """Return a lazily initialized, process-wide spaCy language pipeline.

    Loaded independently of other modules (e.g. ``keyword_extractor``)
    to keep this module self-contained per the pipeline's
    single-responsibility, independent-module design, at the cost of
    each module holding its own cached model instance in memory.

    Returns:
        Language: The shared spaCy pipeline instance for this module.
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model: %s", _SPACY_MODEL_NAME)
        _nlp = spacy.load(_SPACY_MODEL_NAME)
    return _nlp


def extract_entities(text: str) -> list[NamedEntity]:
    """Extract structured named entities from text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` after keyword extraction. Entities are
    deduplicated by exact text and category, with an occurrence count
    recorded for each, and ranked by descending frequency.

    Args:
        text: The cleaned text to extract named entities from.

    Raises:
        NERExtractionError: If ``text`` is empty/whitespace-only.

    Returns:
        list[NamedEntity]: Structured entities, each with its surface
            text, mapped category, and occurrence count, ordered by
            descending frequency.
    """
    if not text or not text.strip():
        raise NERExtractionError("Cannot extract named entities from empty text.")

    nlp = _get_nlp()
    doc = nlp(text[:_MAX_INPUT_CHARACTERS])

    occurrence_counts: Counter[tuple[str, str]] = Counter()
    display_forms: dict[tuple[str, str], str] = {}

    for ent in doc.ents:
        mapped_type = _ENTITY_LABEL_MAP.get(ent.label_)
        if mapped_type is None:
            continue

        surface_text = ent.text.strip()
        if not surface_text:
            continue

        key = (surface_text.lower(), mapped_type)
        occurrence_counts[key] += 1
        display_forms.setdefault(key, surface_text)

    ranked = sorted(
        occurrence_counts.items(),
        key=lambda item: (-item[1], item[0][1], item[0][0]),
    )

    entities = [
        NamedEntity(text=display_forms[key], entity_type=key[1], count=count)
        for key, count in ranked
    ]

    logger.info("Extracted %d unique named entity/entities from text.", len(entities))
    return entities