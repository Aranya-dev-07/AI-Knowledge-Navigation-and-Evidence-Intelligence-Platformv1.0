"""Factual claim extraction for the TrustLens AI Content Understanding Engine.

Single responsibility: given cleaned text, identify sentences that
read as factual, checkable statements and surface them as candidate
claims for later verification elsewhere in the system. Uses spaCy for
sentence segmentation and lightweight linguistic filtering (questions,
missing verbs, first-person opinion markers), then a Hugging Face
Transformers zero-shot classifier to distinguish factual statements
from opinion at the sentence level. This module does NOT verify,
fact-check, or score the truthfulness of any claim \u2014 it only decides
whether a sentence is a *candidate* worth verifying downstream.
"""

import logging

import spacy
from spacy.language import Language
from transformers import Pipeline, pipeline

from engines.content.schemas import Claim

logger = logging.getLogger(__name__)

_SPACY_MODEL_NAME: str = "en_core_web_sm"
_ZERO_SHOT_MODEL_NAME: str = "facebook/bart-large-mnli"
_MAX_INPUT_CHARACTERS: int = 100_000
_MIN_CLAIM_CONFIDENCE: float = 0.6
_MIN_SENTENCE_WORD_COUNT: int = 4
_CLAIM_LABELS: list[str] = ["factual claim", "personal opinion"]
_FACTUAL_LABEL: str = "factual claim"

_OPINION_MARKERS: frozenset[str] = frozenset(
    {
        "i think",
        "i believe",
        "i feel",
        "i suspect",
        "i guess",
        "in my opinion",
        "in my view",
        "personally",
        "it seems to me",
        "we think",
        "we believe",
    }
)

_nlp: Language | None = None
_claim_classifier: Pipeline | None = None


class ClaimExtractionError(Exception):
    """Raised when claim extraction fails."""


def _get_nlp() -> Language:
    """Return a lazily initialized, process-wide spaCy language pipeline.

    Loaded independently of other modules to keep this module
    self-contained per the pipeline's single-responsibility design.

    Returns:
        Language: The shared spaCy pipeline instance for this module.
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model: %s", _SPACY_MODEL_NAME)
        _nlp = spacy.load(_SPACY_MODEL_NAME)
    return _nlp


def _get_claim_classifier() -> Pipeline:
    """Return a lazily initialized, process-wide zero-shot classification pipeline.

    Maintained separately from ``topic_classifier``'s pipeline instance
    to keep this module independently usable, at the cost of caching
    the same underlying model twice in memory if both modules are used.

    Returns:
        Pipeline: The shared zero-shot classification pipeline for
            distinguishing factual claims from opinions.
    """
    global _claim_classifier
    if _claim_classifier is None:
        logger.info("Loading zero-shot classification model: %s", _ZERO_SHOT_MODEL_NAME)
        _claim_classifier = pipeline("zero-shot-classification", model=_ZERO_SHOT_MODEL_NAME)
    return _claim_classifier


def _is_candidate_sentence(sentence_text: str, has_verb: bool) -> bool:
    """Apply lightweight linguistic filtering to discard non-claim sentences.

    Filters out questions, verb-less fragments, overly short sentences,
    and sentences containing common first-person opinion markers
    (e.g. "I think", "in my opinion") before the more expensive
    transformer-based classification step runs.

    Args:
        sentence_text: The sentence text to evaluate.
        has_verb: Whether spaCy identified at least one verb token in
            the sentence.

    Returns:
        bool: ``True`` if the sentence is worth passing to the
            zero-shot classifier, ``False`` if it can be confidently
            discarded already.
    """
    if not sentence_text or sentence_text.endswith("?"):
        return False

    if len(sentence_text.split()) < _MIN_SENTENCE_WORD_COUNT:
        return False

    if not has_verb:
        return False

    lowered = sentence_text.lower()
    if any(marker in lowered for marker in _OPINION_MARKERS):
        return False

    return True


def _collect_candidate_sentences(text: str) -> list[str]:
    """Segment text into sentences and apply pre-classification filtering.

    Args:
        text: The cleaned text to segment.

    Returns:
        list[str]: Sentence texts that survived lightweight linguistic
            filtering and are worth sending to the claim classifier.
    """
    nlp = _get_nlp()
    doc = nlp(text[:_MAX_INPUT_CHARACTERS])

    candidates: list[str] = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        has_verb = any(token.pos_ == "VERB" for token in sent)
        if _is_candidate_sentence(sentence_text, has_verb):
            candidates.append(sentence_text)

    return candidates


def extract_claims(text: str) -> list[Claim]:
    """Extract candidate factual claims from text.

    This is the single public entry point for this module, called by
    ``pipeline.py`` after named entity recognition. Candidate
    sentences are first filtered linguistically (questions, missing
    verbs, opinion markers), then classified by a zero-shot
    transformer model as either a "factual claim" or "personal
    opinion"; only sentences confidently classified as factual claims
    are returned. No verification of claim truthfulness is performed.

    Args:
        text: The cleaned text to extract candidate claims from.

    Raises:
        ClaimExtractionError: If ``text`` is empty/whitespace-only, or
            the underlying classification model fails.

    Returns:
        list[Claim]: Candidate factual claims, each with its sentence
            text and the classifier's confidence, ordered by
            descending confidence. Returns an empty list if no
            sentences qualify as candidate claims.
    """
    if not text or not text.strip():
        raise ClaimExtractionError("Cannot extract claims from empty text.")

    candidate_sentences = _collect_candidate_sentences(text)
    if not candidate_sentences:
        logger.info("No candidate claim sentences found after linguistic filtering.")
        return []

    try:
        classifier = _get_claim_classifier()
        results = classifier(candidate_sentences, candidate_labels=_CLAIM_LABELS, multi_label=False)
    except Exception as exc:
        raise ClaimExtractionError(f"Claim classification failed: {exc}") from exc

    # transformers returns a single dict (not a list) when given exactly
    # one input string; normalize to a list for uniform handling.
    if isinstance(results, dict):
        results = [results]

    claims: list[Claim] = []
    for sentence_text, result in zip(candidate_sentences, results, strict=True):
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        if top_label == _FACTUAL_LABEL and top_score >= _MIN_CLAIM_CONFIDENCE:
            claims.append(Claim(text=sentence_text, confidence=round(top_score, 4)))

    claims.sort(key=lambda claim: claim.confidence, reverse=True)

    logger.info(
        "Extracted %d candidate claim(s) from %d candidate sentence(s).",
        len(claims),
        len(candidate_sentences),
    )
    return claims