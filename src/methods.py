"""Implementation of the 4 confidence calibration methods:
1. Token probability (length-normalized geometric mean of greedy answer tokens)
2. Verbalized confidence (0-100 numerical elicitation)
3. P(True) self-check (next-token probability of True vs False)
4. Self-consistency (agreement across 5 sampled answers at T=0.7)
"""

import math
import re
from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from matcher import normalize_text, strip_answer_prefixes, extract_sciq_letter, match_trivia_qa, match_sciq


def build_qa_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for question answering."""
    if item["dataset"] == "trivia_qa":
        user_prompt = f"Answer the following question in as few words as possible. Output only the answer, nothing else.\nQuestion: {item['question']}\nAnswer:"
    else:  # sciq
        choices = item["choices"]
        user_prompt = (
            f"Question: {item['question']}\n"
            f"(A) {choices['A']}\n"
            f"(B) {choices['B']}\n"
            f"(C) {choices['C']}\n"
            f"(D) {choices['D']}\n"
            f"Answer with only the letter corresponding to the correct answer: (A), (B), (C), or (D).\nAnswer:"
        )

    return [
        {"role": "system", "content": "You are a direct, concise question-answering assistant."},
        {"role": "user", "content": user_prompt},
    ]


def run_greedy_answer_and_token_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 32,
) -> Tuple[str, float]:
    """Generate greedy answer and compute length-normalized token probability."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    gen_token_ids = outputs.sequences[0][input_len:]
    answer_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()

    # Calculate length-normalized token probability
    if len(outputs.scores) == 0:
        return answer_text, 0.5

    log_probs = []
    for step_idx, step_logits in enumerate(outputs.scores):
        if step_idx >= len(gen_token_ids):
            break
        token_id = gen_token_ids[step_idx]
        probs = torch.softmax(step_logits[0], dim=-1)
        p = probs[token_id].item()
        log_probs.append(math.log(max(p, 1e-12)))

    if not log_probs:
        token_prob = 0.5
    else:
        avg_log_p = sum(log_probs) / len(log_probs)
        token_prob = float(min(max(math.exp(avg_log_p), 0.0), 1.0))

    return answer_text, token_prob


def run_verbalized_confidence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    device: torch.device,
) -> Tuple[float, str, bool]:
    """Ask model for verbalized confidence on 0-100 scale."""
    user_prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"How confident are you that this answer is correct? "
        f"Respond with only a number between 0 and 100, where 0 means completely uncertain and 100 means completely certain."
    )
    messages = [
        {"role": "system", "content": "You are an evaluator. Respond with only a single number between 0 and 100."},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    raw_resp = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # Regex parse number between 0 and 100
    m = re.search(r"\b(100(?:\.0+)?|[1-9]?[0-9](?:\.[0-9]+)?)\b", raw_resp)
    if m:
        num = float(m.group(1))
        conf = float(min(max(num / 100.0, 0.0), 1.0))
        return conf, raw_resp, True
    else:
        # Fallback if no number parsed
        return 0.5, raw_resp, False


def get_true_false_token_ids(tokenizer: PreTrainedTokenizer) -> Tuple[List[int], List[int]]:
    """Find token IDs corresponding to True/False variants in tokenizer."""
    true_ids = set()
    false_ids = set()
    for w in ["True", " True", "true", " true"]:
        tokens = tokenizer.encode(w, add_special_tokens=False)
        if len(tokens) == 1:
            true_ids.add(tokens[0])
    for w in ["False", " False", "false", " false"]:
        tokens = tokenizer.encode(w, add_special_tokens=False)
        if len(tokens) == 1:
            false_ids.add(tokens[0])
    return list(true_ids), list(false_ids)


def run_p_true(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    device: torch.device,
    true_ids: List[int],
    false_ids: List[int],
) -> float:
    """Compute P(True) self-check next-token probability."""
    user_prompt = (
        f"Question: {question}\n"
        f"Proposed Answer: {answer}\n"
        f"Is this answer correct? Answer True or False.\nAnswer:"
    )
    messages = [
        {"role": "system", "content": "You are an evaluator. Determine if the proposed answer is correct. Answer True or False."},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    p_true = sum(probs[tid].item() for tid in true_ids)
    p_false = sum(probs[fid].item() for fid in false_ids)

    denom = p_true + p_false
    if denom <= 1e-12:
        return 0.5
    return float(min(max(p_true / denom, 0.0), 1.0))


def run_self_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    greedy_answer: str,
    dataset_type: str,
    device: torch.device,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 32,
) -> Tuple[float, List[str]]:
    """Sample multiple generations and compute agreement with greedy answer."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=num_samples,
        )

    samples = []
    matches = 0

    if dataset_type == "sciq":
        greedy_letter = extract_sciq_letter(greedy_answer)
        for i in range(num_samples):
            seq = outputs[i][input_len:]
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            samples.append(text)
            sample_letter = extract_sciq_letter(text)
            if greedy_letter and sample_letter == greedy_letter:
                matches += 1
            elif not greedy_letter and normalize_text(text) == normalize_text(greedy_answer):
                matches += 1
    else:  # trivia_qa
        norm_greedy = normalize_text(strip_answer_prefixes(greedy_answer))
        for i in range(num_samples):
            seq = outputs[i][input_len:]
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            samples.append(text)
            norm_sample = normalize_text(strip_answer_prefixes(text))
            if norm_greedy and (norm_sample == norm_greedy or match_trivia_qa(text, [greedy_answer])):
                matches += 1

    agreement = float(matches / num_samples)
    return agreement, samples


def evaluate_item_all_methods(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    item: Dict[str, Any],
    device: torch.device,
    true_ids: List[int],
    false_ids: List[int],
) -> Dict[str, Any]:
    """Run greedy answer + all 4 confidence methods on a single question item."""
    messages = build_qa_messages(item)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 1. Greedy answer + Token probability
    answer_text, token_prob = run_greedy_answer_and_token_prob(
        model, tokenizer, prompt_text, device
    )

    # Ground truth correctness
    if item["dataset"] == "trivia_qa":
        is_correct = match_trivia_qa(answer_text, item["aliases"])
        target_str = item["target"]
    else:  # sciq
        is_correct = match_sciq(answer_text, item["correct_letter"], item["correct_text"])
        target_str = f"({item['correct_letter']}) {item['correct_text']}"

    # 2. Verbalized confidence
    conf_verbalized, raw_verbalized, verbalized_ok = run_verbalized_confidence(
        model, tokenizer, item["question"], answer_text, device
    )

    # 3. P(True) self-check
    conf_p_true = run_p_true(
        model, tokenizer, item["question"], answer_text, device, true_ids, false_ids
    )

    # 4. Self-consistency (5 samples)
    conf_consistency, consistency_samples = run_self_consistency(
        model, tokenizer, prompt_text, answer_text, item["dataset"], device, num_samples=5
    )

    # Guarantee no NaNs
    def clean_float(val: float, default: float = 0.5) -> float:
        if val is None or math.isnan(val) or math.isinf(val):
            return default
        return float(min(max(val, 0.0), 1.0))

    return {
        "dataset": item["dataset"],
        "id": item["id"],
        "question": item["question"],
        "target": target_str,
        "greedy_answer": answer_text,
        "is_correct": bool(is_correct),
        "conf_token_prob": clean_float(token_prob),
        "conf_verbalized": clean_float(conf_verbalized),
        "conf_p_true": clean_float(conf_p_true),
        "conf_consistency": clean_float(conf_consistency),
        "raw_verbalized_text": raw_verbalized,
        "verbalized_parsed": bool(verbalized_ok),
        "consistency_samples": consistency_samples,
    }
