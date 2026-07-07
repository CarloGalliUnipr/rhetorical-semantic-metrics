import numpy as np
import pandas as pd
from rhetorical_semantic_metrics.metrics import compute_abstract_metrics

def test_compute_basic_metrics():
    df = pd.DataFrame({
        'sentence_id': [1,2,3,4],
        'label': ['Background','Aim','Results','Conclusion'],
        'embedding': [
            np.array([1.,0.]),
            np.array([1.,0.]),
            np.array([0.,1.]),
            np.array([0.,1.]),
        ]
    })
    m = compute_abstract_metrics(df)
    assert m['n_sentences'] == 4
    assert 'mean_local_similarity' in m
