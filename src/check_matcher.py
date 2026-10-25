"""30-item hand check verification for TriviaQA and SciQ matchers."""

from typing import List, Dict, Any
from matcher import match_trivia_qa, match_sciq


def run_hand_check():
    # 15 hand-crafted test cases for TriviaQA covering typical model responses and edge cases
    trivia_cases = [
        # (prediction, aliases, expected_correct, reason)
        ("Paris", ["Paris", "City of Light"], True, "Exact match"),
        ("paris", ["Paris"], True, "Case insensitivity"),
        ("The answer is: Paris.", ["Paris"], True, "Prefix and punctuation stripping"),
        ("It is Paris", ["Paris"], True, "Leading 'It is'"),
        ("David Seville", ["David Seville", "Ross Bagdasarian"], True, "Multi-word alias match"),
        ("Ross Bagdasarian, Sr.", ["Ross Bagdasarian", "Ross Bagdasarian, Sr."], True, "Punctuation in alias"),
        ("London", ["Paris"], False, "Completely wrong answer"),
        ("Paris, France", ["Paris"], True, "Word boundary phrase match"),
        ("1969", ["1969", "69"], True, "Number match"),
        ("1970", ["1969"], False, "Wrong year"),
        ("Mount Everest", ["Everest", "Mt. Everest"], True, "Abbreviation in aliases"),
        ("The Pacific Ocean", ["Pacific", "Pacific Ocean"], True, "Article stripping"),
        ("Atlantic Ocean", ["Pacific Ocean"], False, "Wrong ocean"),
        ("George Washington", ["George Washington", "Washington"], True, "Famous person alias"),
        ("Thomas Jefferson", ["George Washington"], False, "Wrong person"),
    ]

    # 15 hand-crafted test cases for SciQ covering letter choices, option texts, and edge cases
    sciq_cases = [
        # (prediction, correct_letter, correct_text, expected_correct, reason)
        ("A", "A", "oxidants", True, "Plain capital letter"),
        ("a", "A", "oxidants", True, "Lowercase letter"),
        ("(A)", "A", "oxidants", True, "Letter in parentheses"),
        ("A.", "A", "oxidants", True, "Letter with trailing period"),
        ("Option A", "A", "oxidants", True, "Leading 'Option'"),
        ("The answer is (B)", "B", "photosynthesis", True, "Conversational phrase with letter"),
        ("B: photosynthesis", "B", "photosynthesis", True, "Letter with colon and text"),
        ("oxidants", "A", "oxidants", True, "Fallback to option text match"),
        ("Oxidants.", "A", "oxidants", True, "Fallback text match with casing/punctuation"),
        ("C", "A", "oxidants", False, "Wrong letter choice"),
        ("(D)", "B", "photosynthesis", False, "Wrong letter in parentheses"),
        ("antioxidants", "A", "oxidants", False, "Distractor text fallback (wrong)"),
        ("Option C: residues", "A", "oxidants", False, "Wrong option with text"),
        ("D", "D", "mitochondria", True, "Correct letter D"),
        ("Answer: C", "C", "chloroplast", True, "Explicit 'Answer: C' prefix"),
    ]

    results = []
    total = len(trivia_cases) + len(sciq_cases)
    passed = 0

    print(f"Running 30-item hand check verification...")
    print("=" * 60)
    print("--- 15 TriviaQA Hand Checks ---")
    for i, (pred, aliases, expected, note) in enumerate(trivia_cases, 1):
        actual = match_trivia_qa(pred, aliases)
        ok = (actual == expected)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{i:02d}/15] {status}: pred='{pred}' | expected={expected} actual={actual} ({note})")
        results.append({
            "idx": i,
            "dataset": "trivia_qa",
            "prediction": pred,
            "expected": expected,
            "actual": actual,
            "ok": ok,
            "note": note,
        })

    print("\n--- 15 SciQ Hand Checks ---")
    for j, (pred, letter, text, expected, note) in enumerate(sciq_cases, 1):
        idx = len(trivia_cases) + j
        actual = match_sciq(pred, letter, text)
        ok = (actual == expected)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{idx:02d}/30] {status}: pred='{pred}' (gold={letter}) | expected={expected} actual={actual} ({note})")
        results.append({
            "idx": idx,
            "dataset": "sciq",
            "prediction": pred,
            "expected": expected,
            "actual": actual,
            "ok": ok,
            "note": note,
        })

    print("=" * 60)
    print(f"Summary: {passed}/{total} checks passed.")
    if passed == total:
        print("ALL 30/30 HAND CHECKS PASSED PERFECTLY!")
    else:
        print(f"FAILED: {total - passed} checks failed.")

    return passed == total


if __name__ == "__main__":
    success = run_hand_check()
    exit(0 if success else 1)
