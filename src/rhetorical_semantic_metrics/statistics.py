import numpy as np
import pandas as pd
from scipy import stats

def cohens_d(review_values, clinical_values):
    review = np.asarray(pd.Series(review_values).dropna(), dtype=float)
    clinical = np.asarray(pd.Series(clinical_values).dropna(), dtype=float)
    n1, n2 = len(review), len(clinical)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1 - 1) * review.var(ddof=1) + (n2 - 1) * clinical.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return float((review.mean() - clinical.mean()) / pooled)

def compare_groups(df, metric_col, dataset_col='Dataset', review_label='Review', clinical_label='Clinical'):
    review = df.loc[df[dataset_col] == review_label, metric_col].dropna()
    clinical = df.loc[df[dataset_col] == clinical_label, metric_col].dropna()
    d = cohens_d(review, clinical)
    welch_p = stats.ttest_ind(review, clinical, equal_var=False, nan_policy='omit').pvalue if len(review) > 1 and len(clinical) > 1 else np.nan
    mw_p = stats.mannwhitneyu(review, clinical, alternative='two-sided').pvalue if len(review) > 0 and len(clinical) > 0 else np.nan
    return {
        'Metric': metric_col,
        'Review_mean': review.mean(),
        'Clinical_mean': clinical.mean(),
        'Review_SD': review.std(ddof=1),
        'Clinical_SD': clinical.std(ddof=1),
        'Mean_difference_Review_minus_Clinical': review.mean() - clinical.mean(),
        'Cohens_d': d,
        'Welch_t_p': welch_p,
        'Mann_Whitney_p': mw_p,
        'n_review': len(review),
        'n_clinical': len(clinical),
    }
