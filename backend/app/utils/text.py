"""Text extraction utilities used by evaluation and investigator modules."""

from __future__ import annotations

import re
from typing import Optional


def extract_option_letter(text: Optional[str]) -> Optional[str]:
    """Extract a single A-D option letter from a prediction string.

    Looks for the pattern ``The correct answer is X`` where X ∈ {A, B, C, D}.
    Returns ``None`` if the pattern is not found or *text* is falsy.

    This is the canonical extraction function ported from the experimental
    benchmark engine (``03_benchmark_engine.py``).
    """
    if not text:
        return None
    match = re.search(r"The correct answer is ([A-D])", text)
    return match.group(1) if match else None


def strip_markdown_json(text: str) -> str:
    """Remove ```json fences and extract the first JSON object block from text.

    Protects against LLMs outputting surrounding conversation or conversational text.
    """
    text = text.strip()
    # Strip markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Search for outermost JSON braces if it doesn't start with {
    if not text.startswith("{"):
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]

    return text.strip()

