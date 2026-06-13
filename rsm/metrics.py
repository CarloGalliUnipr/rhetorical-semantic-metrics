"""Rhetorical-semantic metric computation."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from .labeling import RHETORICAL_ROLES, label_sentences_with_llm, maybe_sleep


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def get_spacy_model():
    return spacy.load("en_core_web_sm")


def split_into_sentences(text: str, nlp) -> List[str]:
    doc = nlp(clean_text(text))
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))


def centroid(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors)
    c = np.mean(vectors, axis=0)
    norm = np.linalg.norm(c)
    return c if norm == 0 else c / norm


def mean_pairwise_similarity(vectors: np.ndarray) -> float:
    vectors = np.asarray(vectors)
    if len(vectors) < 2:
        return np.nan
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_sim(vectors[i], vectors[j]))
    return float(np.mean(sims))


def local_similarity(embeddings: np.ndarray) -> np.ndarray:
    sims = [np.nan]
    for i in range(1, len(embeddings)):
        sims.append(cosine_sim(embeddings[i], embeddings[i - 1]))
    return np.array(sims)


def role_indices(df_sent: pd.DataFrame, role: str) -> List[int]:
    return df_sent.index[df_sent["rhetorical_role"] == role].tolist()


def role_vectors(df_sent: pd.DataFrame, embeddings: np.ndarray, role: str) -> Optional[np.ndarray]:
    idx = role_indices(df_sent, role)
    if not idx:
        return None
    return embeddings[idx]


def role_centroid(df_sent: pd.DataFrame, embeddings: np.ndarray, role: str) -> Optional[np.ndarray]:
    vectors = role_vectors(df_sent, embeddings, role)
    if vectors is None or len(vectors) == 0:
        return None
    return centroid(vectors)


def combined_role_centroid(df_sent: pd.DataFrame, embeddings: np.ndarray, roles: List[str]) -> Optional[np.ndarray]:
    idx = df_sent.index[df_sent["rhetorical_role"].isin(roles)].tolist()
    if not idx:
        return None
    return centroid(embeddings[idx])


def centroid_similarity_between_roles(df_sent: pd.DataFrame, embeddings: np.ndarray, role_a: str, role_b: str) -> float:
    ca = role_centroid(df_sent, embeddings, role_a)
    cb = role_centroid(df_sent, embeddings, role_b)
    if ca is None or cb is None:
        return np.nan
    return cosine_sim(ca, cb)


def centroid_similarity_combined_roles(
    df_sent: pd.DataFrame,
    embeddings: np.ndarray,
    roles_a: List[str],
    roles_b: List[str],
) -> float:
    ca = combined_role_centroid(df_sent, embeddings, roles_a)
    cb = combined_role_centroid(df_sent, embeddings, roles_b)
    if ca is None or cb is None:
        return np.nan
    return cosine_sim(ca, cb)


def mean_within_role_similarity(df_sent: pd.DataFrame, embeddings: np.ndarray, role: str) -> float:
    vectors = role_vectors(df_sent, embeddings, role)
    if vectors is None or len(vectors) < 2:
        return np.nan
    return mean_pairwise_similarity(vectors)


def leave_one_out_role_anchor_similarity(df_sent: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    loo_sims = np.full(len(df_sent), np.nan)
    for role in RHETORICAL_ROLES:
        idx = role_indices(df_sent, role)
        if len(idx) < 2:
            continue
        for i in idx:
            other_idx = [j for j in idx if j != i]
            if other_idx:
                anchor = centroid(embeddings[other_idx])
                loo_sims[i] = cosine_sim(embeddings[i], anchor)
    return loo_sims


def check_required_roles(labels: Sequence[str], required_roles: Sequence[str]) -> None:
    """Raise a descriptive error if required rhetorical roles are absent."""
    observed_roles = set(labels)
    missing_roles = set(required_roles) - observed_roles
    role_counts = pd.Series(labels).value_counts().to_dict()

    if missing_roles:
        raise ValueError(
            f"Missing required rhetorical roles: {sorted(list(missing_roles))} | "
            f"Observed roles: {sorted(list(observed_roles))} | "
            f"Role counts: {role_counts}"
        )


def analyze_single_abstract(
    abstract_text: str,
    pmid: Optional[str],
    title: Optional[str],
    dataset: Optional[str],
    nlp,
    embedding_model: SentenceTransformer,
    openai_client,
    llm_model_name: str,
    min_sentences: int = 5,
    required_roles: Sequence[str] = ("Methods", "Results"),
    sleep_between_llm_calls: float = 0.5,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    sentences = split_into_sentences(abstract_text, nlp)
    if len(sentences) < min_sentences:
        raise ValueError(f"Too few sentences for reliable analysis: {len(sentences)}")

    labels, confidences, raw_response = label_sentences_with_llm(
        sentences,
        client=openai_client,
        model_name=llm_model_name,
        verbose=verbose,
    )
    maybe_sleep(sleep_between_llm_calls)

    check_required_roles(labels, required_roles)
    observed_roles = set(labels)

    df_sent = pd.DataFrame({
        "PMID": pmid,
        "Title": title,
        "Dataset": dataset,
        "sentence_id": range(1, len(sentences) + 1),
        "sentence": sentences,
        "rhetorical_role": labels,
        "label_confidence": confidences,
    })

    embeddings = embedding_model.encode(
        df_sent["sentence"].tolist(),
        normalize_embeddings=True,
    )

    df_sent["local_similarity_to_previous_sentence"] = local_similarity(embeddings)

    opening_idx = df_sent.index[df_sent["rhetorical_role"].isin(["Background", "Aim"])].tolist()
    if len(opening_idx) == 0:
        opening_idx = list(range(min(2, len(df_sent))))
    opening_anchor = centroid(embeddings[opening_idx])

    df_sent["similarity_to_opening_anchor"] = [cosine_sim(vec, opening_anchor) for vec in embeddings]
    df_sent["similarity_to_own_role_anchor_loo"] = leave_one_out_role_anchor_similarity(df_sent, embeddings)

    df_sent["previous_role"] = df_sent["rhetorical_role"].shift(1)
    df_sent["same_role_as_previous"] = df_sent["rhetorical_role"] == df_sent["previous_role"]

    threshold = (
        np.nanmean(df_sent["local_similarity_to_previous_sentence"])
        - np.nanstd(df_sent["local_similarity_to_previous_sentence"])
    )
    df_sent["low_local_similarity_flag"] = df_sent["local_similarity_to_previous_sentence"] < threshold
    df_sent["within_role_low_similarity_flag"] = (
        df_sent["low_local_similarity_flag"] & df_sent["same_role_as_previous"]
    )

    summary = {
        "PMID": pmid,
        "Title": title,
        "Dataset": dataset,
        "n_sentences": len(df_sent),
        "mean_label_confidence": np.nanmean(df_sent["label_confidence"]),
        "mean_local_similarity": np.nanmean(df_sent["local_similarity_to_previous_sentence"]),
        "min_local_similarity": np.nanmin(df_sent["local_similarity_to_previous_sentence"]),
        "local_similarity_volatility": np.nanstd(df_sent["local_similarity_to_previous_sentence"]),
        "local_discontinuity_burden": np.nanmean(df_sent["low_local_similarity_flag"]),
        "within_role_discontinuity_burden": np.nanmean(df_sent["within_role_low_similarity_flag"]),
        "mean_similarity_to_opening_anchor": np.nanmean(df_sent["similarity_to_opening_anchor"]),
        "final_similarity_to_opening_anchor": np.nanmean(df_sent["similarity_to_opening_anchor"].tail(2)),
        "mean_similarity_to_own_role_anchor_loo": np.nanmean(df_sent["similarity_to_own_role_anchor_loo"]),
        "observed_roles": "; ".join(sorted(list(observed_roles))),
    }

    for role in RHETORICAL_ROLES:
        summary[f"n_{role}"] = int((df_sent["rhetorical_role"] == role).sum())
        summary[f"within_{role}_similarity"] = mean_within_role_similarity(df_sent, embeddings, role)

    summary["conclusion_to_aim_alignment"] = centroid_similarity_between_roles(df_sent, embeddings, "Conclusion", "Aim")
    summary["conclusion_to_results_alignment"] = centroid_similarity_between_roles(df_sent, embeddings, "Conclusion", "Results")
    summary["results_to_methods_alignment"] = centroid_similarity_between_roles(df_sent, embeddings, "Results", "Methods")
    summary["methods_to_aim_alignment"] = centroid_similarity_between_roles(df_sent, embeddings, "Methods", "Aim")
    summary["results_to_aim_alignment"] = centroid_similarity_between_roles(df_sent, embeddings, "Results", "Aim")
    summary["conclusion_to_background_plus_aim_alignment"] = centroid_similarity_combined_roles(
        df_sent,
        embeddings,
        ["Conclusion"],
        ["Background", "Aim"],
    )
    summary["conclusion_plus_future_to_background_plus_aim_alignment"] = centroid_similarity_combined_roles(
        df_sent,
        embeddings,
        ["Conclusion", "Limitation/Future"],
        ["Background", "Aim"],
    )

    return df_sent, summary
