"""Engines package for TrustLens AI backend.

Top-level namespace for TrustLens AI's processing engines. Currently
exposes the Content Understanding Engine (``engines.content``); future
engines (e.g. trust scoring, fact-checking) will be added as sibling
subpackages and exposed here in the same manner.
"""

from engines import content

__all__ = ["content"]