# Rhetorical-Semantic Metrics for Scientific Abstracts

This repository contains code for building a PubMed pilot corpus and computing rhetorical-semantic coherence metrics for scientific abstracts. The workflow was designed for the paper project **Measuring Scientific Narrative Coherence: A Rhetorical-Semantic Metric Framework for Research Abstracts**.

## What the pipeline does

1. Queries PubMed for two comparable periodontal-disease abstract datasets:
   - reviews, n = 50
   - clinical studies, n = 50
2. Splits each abstract into sentences.
3. Uses an OpenAI model to assign rhetorical roles:
   - Background
   - Aim
   - Methods
   - Results
   - Conclusion
   - Limitation/Future
4. Encodes sentences using a sentence-transformer model.
5. Computes abstract-level and sentence-level rhetorical-semantic metrics.
6. Exports tables and plots comparing reviews and clinical studies.

## Repository structure

```text
rhetorical_semantic_metrics/
├── config.yaml
├── requirements.txt
├── rsm/
│   ├── pubmed.py
│   ├── labeling.py
│   ├── metrics.py
│   └── analysis.py
├── scripts/
│   ├── build_pubmed_datasets.py
│   └── run_analysis.py
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
    ├── figures/
    └── tables/
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration

Edit `config.yaml` before running.

At minimum, set:

```yaml
pubmed:
  email: your.real.email@example.com
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key"
```

In Colab:

```python
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
```

## Step 1: Build PubMed datasets

```bash
python scripts/build_pubmed_datasets.py
```

Outputs:

```text
data/raw/periodontal_reviews_50.csv
data/raw/periodontal_clinical_50.csv
data/raw/periodontal_reviews_vs_clinical_100.csv
```

## Step 2: Run rhetorical-semantic analysis

```bash
python scripts/run_analysis.py
```

Outputs:

```text
outputs/all_sentence_level_metrics.csv
outputs/all_abstract_level_metrics.csv
outputs/tables/dataset_level_metric_summary_mean_sd_sem.csv
outputs/tables/table_mean_sd_reviews_vs_clinical.csv
outputs/tables/statistical_comparison_reviews_vs_clinical.csv
outputs/figures/barplot_reviews_vs_clinical_core_metrics.png
outputs/figures/boxplot_*.png
```

## Main metrics

The main metric families are:

- **Local semantic continuity**: adjacent-sentence cosine similarity.
- **Local discontinuity burden**: proportion of low-similarity transitions.
- **Within-role semantic consistency**: pairwise similarity among sentences assigned to the same rhetorical role.
- **Leave-one-out role-anchor similarity**: sentence-to-role consistency without self-inflation.
- **Conclusion-to-aim alignment**: rhetorical closure.
- **Conclusion-to-results alignment**: interpretive grounding.
- **Results-to-methods alignment**: evidential linkage.

## Notes

- The code does not assess scientific quality or methodological validity.
- LLM labels should be manually audited for the final paper, at least on a validation subset.
- PubMed results may change over time. For reproducibility, archive the exact generated CSV files used in the analysis.

## Citation

If you use this code, cite the associated paper once published.
