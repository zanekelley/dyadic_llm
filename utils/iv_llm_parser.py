"""
<llm_parser> This script provides utilities for parsing, normalizing, and exporting
log-probability outputs from structured LLM decisions.

It converts token-level log probabilities returned by an OpenAI chat model
into interpretable class-level probabilities, handling JSON-schema–constrained
outputs and common tokenization artifacts (e.g., quoted or split tokens).
The primary function, `parse_llm_decision`, extracts the predicted class,
aggregates log probabilities for competing classes, and normalizes them
via a softmax transformation.

Key features:
- Aggregates token-level log probabilities into class-level scores
- Robust to JSON schema tokenization (e.g., `"male"`, `male"`)
- Computes normalized probabilities from log probabilities
- Supports extensible structured question types (currently gender)
- Flattens parsed results into a tidy tabular format for analysis
- Exports results to CSV for downstream statistical workflows
"""

import math
import pandas as pd

def logprobs_to_percentages(logprob_dict):
    """Convert {label: logprob} → {label: percentage} using softmax."""
    max_log = max(logprob_dict.values())
    exps = {k: math.exp(v - max_log) for k, v in logprob_dict.items()}
    total = sum(exps.values())
    return {k: exps[k] / total for k in exps}

def parse_llm_decision(result: dict, structured_question_type: str):
    """
    result: {
        "content": {"gender": "male"},
        "logprobs": [ChatCompletionTokenLogprob, ...]
    }
    """
    content = result["content"]
    logprobs = result["logprobs"] or []

    if structured_question_type == "gender":
        predicted_value = content["gender"] # Only extracts the male or female chunk from the dictionary
        classes = ["male", "female"]

    else:
        raise ValueError(f"Unsupported type: {structured_question_type}")

    # 1. Find the token-level logprob of the predicted class
    matching_tokens = [
        tok for tok in logprobs
        if tok.token.strip('" ') == predicted_value # looks for predicted value even if it has miscellaneous quotes
                                                    # takes 'male' or '"male' or 'male"'
    ]

    if not matching_tokens:
        predicted_logprob = None
    else:
        # sum because JSON schema sometimes tokenizes like `"male"` or '"male'
        predicted_logprob = sum(tok.logprob for tok in matching_tokens)

    # 2. Extract top_logprobs for ALL competing classes
    class_logprobs = {}

    # Locate the FIRST matching token (mirrors TS `.find`)
    token_obj = next(iter(matching_tokens), None)

    if token_obj and token_obj.top_logprobs:
        # top_logprobs is a list of TopLogprob objects
        for top in token_obj.top_logprobs:
            clean_token = top.token.strip('" ')
            if clean_token in classes:
                class_logprobs[clean_token] = top.logprob

    # Fallback: if top_logprobs is missing, use only the predicted logprob
    if not class_logprobs and predicted_logprob is not None:
        class_logprobs[predicted_value] = predicted_logprob

    # 3. Convert to probabilities
    percentages = (
        logprobs_to_percentages(class_logprobs)
        if class_logprobs else None
    )

    return {
        "predicted_value": predicted_value,
        "predicted_logprob": predicted_logprob,
        "class_logprobs": class_logprobs,     # logprobs for all classes
        "percentages": percentages            # normalized 0–1 probabilities
    }


def results_dict_to_csv(results_dict, output_path="outputs/output.csv"):
    rows = []

    for key_tuple, result in results_dict.items():
        # unpack tuple (e.g., turn_number, speaker, text)
        dyad, speaker, text = key_tuple

        # flatten nested dicts
        row = {
            "dyad_id": dyad,
            "speaker": speaker,
            "text": text,
            "predicted_value": result["predicted_value"],
            "predicted_logprob": result["predicted_logprob"],
            "female_logprob": result["class_logprobs"]["female"],
            "male_logprob": result["class_logprobs"]["male"],
            "female_pct": result["percentages"]["female"],
            "male_pct": result["percentages"]["male"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df