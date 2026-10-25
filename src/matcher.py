"""Answer matching logic for TriviaQA and SciQ."""

import re
import string
from typing import List, Optional


def normalize_text(text: str) -> str:
    """Standard QA normalization: lowercase, remove punctuation, remove articles, collapse spaces."""
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = "".join(ch if ch not in set(string.punctuation) else " " for ch in text)

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse whitespace
    return " ".join(text.split())


def strip_answer_prefixes(text: str) -> str:
    """Strip common conversational prefixes models output before the actual answer."""
    clean = text.strip()
    prefixes = [
        r"^answer\s*:\s*",
        r"^the\s+answer\s+is\s*:?\s*",
        r"^it\s+is\s*:?\s*",
        r"^they\s+are\s*:?\s*",
        r"^it\'?s\s*:?\s*",
    ]
    for pattern in prefixes:
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE).strip()
    return clean


def match_trivia_qa(prediction: str, ground_truth_aliases: List[str]) -> bool:
    """Check if prediction matches any alias in ground_truth_aliases."""
    if not prediction or not ground_truth_aliases:
        return False

    cleaned_pred = strip_answer_prefixes(prediction)
    norm_pred = normalize_text(cleaned_pred)

    if not norm_pred:
        return False

    norm_aliases = [normalize_text(a) for a in ground_truth_aliases]
    norm_aliases = [a for a in norm_aliases if a]

    # 1. Exact match against any normalized alias
    for alias in norm_aliases:
        if norm_pred == alias:
            return True

    # 2. Word-boundary phrase match in prediction for non-trivial aliases
    # (avoid single letters or 1-2 char words triggering false positive substrings)
    for alias in norm_aliases:
        if len(alias) >= 3 or len(alias.split()) > 1:
            escaped_alias = re.escape(alias)
            if re.search(r"\b" + escaped_alias + r"\b", norm_pred):
                return True

    return False


def extract_sciq_letter(prediction: str) -> Optional[str]:
    """Extract multiple-choice letter (A, B, C, or D) from model output."""
    if not prediction:
        return None

    clean = prediction.strip()

    # Pattern 1: Starts with option letter, e.g. "(A)", "A.", "A)", "A:" or just "A"
    m = re.match(r"^\(?([A-D])\)?(?:[\.\:\)\s]|$)", clean, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: Explicit "Option X" or "Answer: X"
    m = re.search(r"\b(?:option|answer\s*is|choice)\s*:?\s*\(?([A-D])\)?\b", clean, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 3: First isolated capital letter A-D
    m = re.search(r"\b([A-D])\b", clean)
    if m:
        return m.group(1).upper()

    return None


def match_sciq(prediction: str, correct_letter: str, correct_text: Optional[str] = None) -> bool:
    """Check if prediction matches the correct letter or option text."""
    if not prediction or not correct_letter:
        return False

    extracted = extract_sciq_letter(prediction)
    if extracted and extracted == correct_letter.strip().upper():
        return True

    # Fallback: check if prediction text directly matches the correct option text
    if correct_text:
        norm_pred = normalize_text(strip_answer_prefixes(prediction))
        norm_correct = normalize_text(correct_text)
        if norm_pred and norm_pred == norm_correct:
            return True

    return False
