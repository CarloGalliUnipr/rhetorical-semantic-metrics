import numpy as np
import pandas as pd

def full_sentence_shuffle(df, random_state=42):
    rng = np.random.default_rng(random_state)
    out = []
    for _, g in df.groupby('PMID'):
        g = g.copy()
        g['sentence_id'] = rng.permutation(g['sentence_id'].values)
        out.append(g.sort_values('sentence_id'))
    return pd.concat(out, ignore_index=True)

def within_role_shuffle(df, random_state=42, label_col='label'):
    rng = np.random.default_rng(random_state)
    out = []
    for _, g in df.groupby('PMID'):
        g = g.copy()
        for _, idx in g.groupby(label_col).groups.items():
            shuffled = rng.permutation(g.loc[idx, 'sentence_id'].values)
            g.loc[idx, 'sentence_id'] = shuffled
        out.append(g.sort_values('sentence_id'))
    return pd.concat(out, ignore_index=True)

def random_labels_preserving_counts(df, random_state=42, label_col='label'):
    rng = np.random.default_rng(random_state)
    out = []
    for _, g in df.groupby('PMID'):
        g = g.copy()
        g[label_col] = rng.permutation(g[label_col].values)
        out.append(g)
    return pd.concat(out, ignore_index=True)

def methods_results_label_swap(df, label_col='label'):
    g = df.copy()
    g[label_col] = g[label_col].replace({'Methods': '__TMP__', 'Results': 'Methods', '__TMP__': 'Results'})
    return g
