import json
import re

ALLOWED_LABELS = ["Background", "Aim", "Methods", "Results", "Conclusion", "Limitation/Future"]

SYSTEM_MESSAGE = "You are a strict rhetorical-role classifier for scientific abstracts. Return valid JSON only."

USER_PROMPT_TEMPLATE = """Classify each sentence of a scientific abstract into exactly one rhetorical role.

Allowed labels:
- Background
- Aim
- Methods
- Results
- Conclusion
- Limitation/Future

Definitions:
- Background: context, rationale, problem statement.
- Aim: study objective, hypothesis, research question.
- Methods: design, sample, procedures, measurements, analysis, recruitment, databases, experiments.
- Results: findings, observations, outcomes, statistics.
- Conclusion: interpretation, implications, recommendations.
- Limitation/Future: limitations, future work, validation needs.

Return only valid JSON in this format:
[
  {{"sentence_id": 1, "label": "Background", "confidence": 0.95}}
]

Sentences:
{sentence_block}
"""

def build_sentence_block(sentences):
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

def build_user_prompt(sentences):
    return USER_PROMPT_TEMPLATE.format(sentence_block=build_sentence_block(sentences))

def parse_llm_json_response(raw_response, n_sentences, allowed_labels=ALLOWED_LABELS):
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    parsed = json.loads(cleaned)
    labels = [None] * n_sentences
    confidences = [None] * n_sentences
    for item in parsed:
        sid = int(item["sentence_id"])
        label = str(item["label"]).strip()
        confidence = float(item.get("confidence", 1.0))
        if label not in allowed_labels:
            raise ValueError(f"Invalid label returned: {label}")
        if sid < 1 or sid > n_sentences:
            raise ValueError(f"Invalid sentence_id returned: {sid}")
        labels[sid - 1] = label
        confidences[sid - 1] = confidence
    if any(label is None for label in labels):
        missing = [i + 1 for i, label in enumerate(labels) if label is None]
        raise ValueError(f"Missing labels for sentences: {missing}")
    return labels, confidences
