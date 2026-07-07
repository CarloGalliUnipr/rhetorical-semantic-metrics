#!/usr/bin/env python
"""Validate that key corrected release outputs are present and internally consistent."""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    'data_public/primary_statistics/statistical_comparison_reviews_vs_clinical.csv',
    'data_public/primary_statistics/role_distribution_mean_sentence_counts.csv',
    'data_public/processed_abstract_level/all_abstract_level_metrics.csv',
    'outputs/tables/appendix_B_table_B1_corrected.csv',
    'outputs/tables/table3_boundary_summary.csv',
]

def main():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print('Missing required files:')
        for p in missing:
            print(' -', p)
        sys.exit(1)

    b1 = pd.read_csv(ROOT / 'outputs/tables/appendix_B_table_B1_corrected.csv')
    of = b1[b1['Metric'].str.contains('Opening-frame', regex=False)]
    assert not of.empty, 'Opening-frame row missing from corrected Table B1.'
    assert abs(float(of.iloc[0]['Role-aware d']) - 0.201) < 1e-6, 'Opening-frame d should be 0.201.'
    assert 'Not separable' in of.iloc[0]['Interpretation'], 'Opening-frame interpretation should be not separable from position.'
    print('Release validation passed.')

if __name__ == '__main__':
    main()
