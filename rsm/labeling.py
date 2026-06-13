"""LLM-based rhetorical-role classification."""

from __future__ import annotations

import json
import re
import time
from typing import List, Tuple

from openai import OpenAI

RHETORICAL_ROLES = [
    "Background",
    "Aim",
    "Methods",
    "Results",
    "Conclusion",
    "Limitation/Future",
]
ALLOWED_LABELS = set(RHETORICAL_ROLES)


def _clean_json_response(raw_response: str) -> str:
    """Remove common Markdown wrappers around JSON responses."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def label_sentences_with_llm(
    sentences: List[str],
    client: OpenAI,
    model_name: str = "gpt-5.4-mini",
    verbose: bool = False,
) -> Tuple[List[str], List[float], str]:
    """Assign one rhetorical role and one confidence score to each sentence."""
    sentence_block = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(sentences)])

    prompt = f"""
You are an expert in scientific writing.

Your task is to classify each sentence of a scientific abstract.

Use exactly one of the following labels:

Background
Aim
Methods
Results
Conclusion
Limitation/Future

Definitions:

Background:
Context, rationale, problem statement.

Aim:
Study objective, hypothesis, research question.

Methods:
Design, sample, procedures, measurements, analysis, recruitment, databases, experiments.

Results:
Findings, observations, outcomes, statistics, evidence summaries.

Conclusion:
Interpretation, implications, recommendations, take-home statements.

Limitation/Future:
Limitations, future work, validation needs.

Return ONLY valid JSON.

Format:
[
  {{
    "sentence_id": 1,
    "label": "Background",
    "confidence": 0.95
  }}
]

Sentences:

{sentence_block}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict rhetorical-role classifier for scientific abstracts. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_response = response.choices[0].message.content.strip()
    if verbose:
        print(raw_response)

    parsed = json.loads(_clean_json_response(raw_response))

    labels = [None] * len(sentences)
    confidences = [None] * len(sentences)

    for item in parsed:
        sid = int(item["sentence_id"])
        label = item["label"]
        confidence = float(item.get("confidence", 1.0))

        if label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid label returned: {label}")
        if sid < 1 or sid > len(sentences):
            raise ValueError(f"Invalid sentence_id returned: {sid}")

        labels[sid - 1] = label
        confidences[sid - 1] = confidence

    missing = [i + 1 for i, x in enumerate(labels) if x is None]
    if missing:
        raise ValueError(f"Missing labels for sentences: {missing}")

    return labels, confidences, raw_response


def maybe_sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds)
