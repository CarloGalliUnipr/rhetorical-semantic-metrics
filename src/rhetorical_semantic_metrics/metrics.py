import numpy as np
import pandas as pd
from itertools import combinations

ROLE_SET = ["Background", "Aim", "Methods", "Results", "Conclusion", "Limitation/Future"]

def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0 or np.isnan(norm):
        return v * np.nan
    return v / norm

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)
    return float(np.dot(a, b))

def role_centroids(sentence_df: pd.DataFrame, embedding_col: str = "embedding", label_col: str = "label") -> dict:
    centroids = {}
    for role, group in sentence_df.groupby(label_col):
        vecs = np.vstack(group[embedding_col].values)
        centroids[role] = normalize_vector(vecs.mean(axis=0))
    return centroids

def mean_local_similarity(sentence_df: pd.DataFrame, embedding_col: str = "embedding") -> float:
    df = sentence_df.sort_values("sentence_id")
    values = []
    embs = list(df[embedding_col].values)
    for i in range(1, len(embs)):
        values.append(cosine(embs[i], embs[i-1]))
    return float(np.nanmean(values)) if values else np.nan

def min_local_similarity(sentence_df: pd.DataFrame, embedding_col: str = "embedding") -> float:
    df = sentence_df.sort_values("sentence_id")
    values = []
    embs = list(df[embedding_col].values)
    for i in range(1, len(embs)):
        values.append(cosine(embs[i], embs[i-1]))
    return float(np.nanmin(values)) if values else np.nan

def local_similarity_volatility(sentence_df: pd.DataFrame, embedding_col: str = "embedding") -> float:
    df = sentence_df.sort_values("sentence_id")
    values = []
    embs = list(df[embedding_col].values)
    for i in range(1, len(embs)):
        values.append(cosine(embs[i], embs[i-1]))
    return float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan

def within_role_similarity(sentence_df: pd.DataFrame, role: str, embedding_col: str = "embedding", label_col: str = "label") -> float:
    subset = sentence_df[sentence_df[label_col] == role]
    if len(subset) < 2:
        return np.nan
    vecs = list(subset[embedding_col].values)
    sims = [cosine(vecs[i], vecs[j]) for i, j in combinations(range(len(vecs)), 2)]
    return float(np.nanmean(sims)) if sims else np.nan

def cross_role_alignment(sentence_df: pd.DataFrame, role_a: str, role_b: str, embedding_col: str = "embedding", label_col: str = "label") -> float:
    centroids = role_centroids(sentence_df, embedding_col, label_col)
    if role_a not in centroids or role_b not in centroids:
        return np.nan
    return cosine(centroids[role_a], centroids[role_b])

def _combined_centroid(sentence_df: pd.DataFrame, roles: list[str], embedding_col: str = "embedding", label_col: str = "label"):
    subset = sentence_df[sentence_df[label_col].isin(roles)]
    if subset.empty:
        return None
    return normalize_vector(np.vstack(subset[embedding_col].values).mean(axis=0))

def opening_frame_return(sentence_df: pd.DataFrame, embedding_col: str = "embedding", label_col: str = "label", broad_final: bool = False, fallback_opening: bool = True) -> float:
    opening = _combined_centroid(sentence_df, ["Background", "Aim"], embedding_col, label_col)
    if opening is None and fallback_opening:
        df = sentence_df.sort_values("sentence_id")
        n = min(2, len(df))
        opening = normalize_vector(np.vstack(df.head(n)[embedding_col].values).mean(axis=0))
    final_roles = ["Conclusion", "Limitation/Future"] if broad_final else ["Conclusion"]
    final = _combined_centroid(sentence_df, final_roles, embedding_col, label_col)
    if opening is None or final is None:
        return np.nan
    return cosine(final, opening)

def compute_abstract_metrics(sentence_df: pd.DataFrame, embedding_col: str = "embedding", label_col: str = "label") -> dict:
    metrics = {
        "n_sentences": len(sentence_df),
        "mean_local_similarity": mean_local_similarity(sentence_df, embedding_col),
        "min_local_similarity": min_local_similarity(sentence_df, embedding_col),
        "local_similarity_volatility": local_similarity_volatility(sentence_df, embedding_col),
    }
    for role in ROLE_SET:
        metrics[f"n_{role}"] = int((sentence_df[label_col] == role).sum())
        metrics[f"within_{role}_similarity"] = within_role_similarity(sentence_df, role, embedding_col, label_col)
    metrics["conclusion_to_results_alignment"] = cross_role_alignment(sentence_df, "Conclusion", "Results", embedding_col, label_col)
    metrics["results_to_methods_alignment"] = cross_role_alignment(sentence_df, "Results", "Methods", embedding_col, label_col)
    metrics["methods_to_aim_alignment"] = cross_role_alignment(sentence_df, "Methods", "Aim", embedding_col, label_col)
    metrics["results_to_aim_alignment"] = cross_role_alignment(sentence_df, "Results", "Aim", embedding_col, label_col)
    metrics["conclusion_to_aim_alignment"] = cross_role_alignment(sentence_df, "Conclusion", "Aim", embedding_col, label_col)
    metrics["conclusion_to_background_plus_aim_alignment"] = opening_frame_return(sentence_df, embedding_col, label_col, broad_final=False)
    metrics["conclusion_plus_future_to_background_plus_aim_alignment"] = opening_frame_return(sentence_df, embedding_col, label_col, broad_final=True)
    return metrics
