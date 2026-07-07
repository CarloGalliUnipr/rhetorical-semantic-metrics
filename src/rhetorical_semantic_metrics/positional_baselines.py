import numpy as np
import pandas as pd
from .metrics import cosine, normalize_vector

def assign_position_bins(sentence_df, n_bins=3):
    df = sentence_df.copy().sort_values(['PMID', 'sentence_id'])
    def bin_one(g):
        n = len(g)
        bins = []
        for k in range(n):
            b = int(np.floor(k * n_bins / n)) + 1
            b = min(b, n_bins)
            bins.append(b)
        g[f'pos_bin_{n_bins}'] = bins
        return g
    return df.groupby('PMID', group_keys=False).apply(bin_one)

def segment_centroid(group, bin_col, bin_id, embedding_col='embedding'):
    subset = group[group[bin_col] == bin_id]
    if subset.empty:
        return None
    return normalize_vector(np.vstack(subset[embedding_col].values).mean(axis=0))

def centroid_alignment_between_bins(group, bin_col, bin_a, bin_b, embedding_col='embedding'):
    a = segment_centroid(group, bin_col, bin_a, embedding_col)
    b = segment_centroid(group, bin_col, bin_b, embedding_col)
    if a is None or b is None:
        return np.nan
    return cosine(a, b)
