"""Smoke test script for S3: runs all 4 confidence methods on n=5 items locally."""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import load_trivia_qa_sample, load_sciq_sample
from methods import evaluate_item_all_methods, get_true_false_token_ids

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_FILE = RESULTS_DIR / "smoke_results.jsonl"


def run_smoke_test(n_items: int = 5):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    true_ids, false_ids = get_true_false_token_ids(tokenizer)
    print(f"Token IDs - True: {true_ids}, False: {false_ids}")

    # Load samples: 3 TriviaQA + 2 SciQ (total 5)
    trivia_items = load_trivia_qa_sample()[:3]
    sciq_items = load_sciq_sample()[:2]
    items = trivia_items + sciq_items

    print(f"\nRunning evaluation on {len(items)} smoke-test items...")
    records = []
    has_nan = False

    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] Evaluating {item['dataset']}: {item['id']}...")
        rec = evaluate_item_all_methods(model, tokenizer, item, device, true_ids, false_ids)
        rec["model_id"] = MODEL_ID

        # Verify no NaN or Inf
        conf_keys = ["conf_token_prob", "conf_verbalized", "conf_p_true", "conf_consistency"]
        for k in conf_keys:
            val = rec[k]
            if val is None or not (0.0 <= val <= 1.0):
                print(f"ERROR: Invalid value for {k}: {val}")
                has_nan = True

        print(
            f"   Answer: '{rec['greedy_answer']}' | Correct: {rec['is_correct']} | "
            f"TokenP: {rec['conf_token_prob']:.3f} | Verb: {rec['conf_verbalized']:.3f} | "
            f"P(True): {rec['conf_p_true']:.3f} | Consist: {rec['conf_consistency']:.3f}"
        )
        records.append(rec)

    # Save to JSONL
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(records)} records to {OUT_FILE}")
    print(f"Verification: No NaN = {not has_nan}, Record count = {len(records)}")

    assert not has_nan, "Found NaN or out-of-range confidence values!"
    assert len(records) == n_items, f"Expected {n_items} records, got {len(records)}"
    print("SMOKE TEST PASSED COMPLETELY!")
    return OUT_FILE


if __name__ == "__main__":
    run_smoke_test(n_items=5)
