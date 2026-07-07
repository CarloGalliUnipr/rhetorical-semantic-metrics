# Rhetorical-Semantic Metrics for Scientific Abstract Organization

This repository contains the code and processed outputs for a role-aware rhetorical-semantic framework for scientific abstracts.

The pipeline segments abstracts into sentences, assigns sentence-level rhetorical roles, computes sentence embeddings, derives local-continuity, within-role, cross-role, and opening-frame metrics, and evaluates these metrics against positional, lexical, sentence-count, embedding-model, and perturbation controls.

The repository is intended for methodological reuse and reproducibility. It is **not** intended to provide automated abstract-quality scores or stand-alone genre classification.

## Associated manuscript

**Beyond Semantic Drift: Testing Role-Aware Semantic Metrics for Scientific Abstract Organization**

Pilot corpus: 100 PubMed periodontics abstracts, including 50 systematic review/meta-analysis abstracts and 50 clinical-study abstracts.

## Main methodological idea

Scientific abstracts are treated as structured rhetorical-semantic trajectories rather than as undifferentiated sentence sequences. Each sentence is assigned to one of six broad rhetorical roles: Background, Aim, Methods, Results, Conclusion, or Limitation/Future.

Sentence embeddings are then used to compute four metric families:

1. **Local continuity**: adjacent sentence-to-sentence similarity.
2. **Within-role consistency**: semantic compactness within a rhetorical role.
3. **Cross-role alignment**: centroid similarity between rhetorical components.
4. **Opening-frame return**: similarity between the final interpretive component and the opening Background/Aim frame.

## Important scope statement

The current analysis supports a bounded claim: role-aware metrics add interpretive structure by attaching similarity patterns to named communicative functions. They do not show general discriminative superiority over simpler positional or lexical baselines.

## Repository structure

```text
src/rhetorical_semantic_metrics/    Reusable Python modules
scripts/                            Numbered pipeline scripts
notebooks/                          Clean walkthrough notebook
data_public/                        Shareable redacted data and frozen outputs
outputs/figures/                    Manuscript-ready figures
outputs/tables/                     Manuscript-ready tables
docs/                               Method notes and data dictionary
examples/                           Small demo input
tests/                              Lightweight consistency tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/13_validate_release_outputs.py
python scripts/12_generate_manuscript_figures_tables.py
```

## Reproducibility levels

This repository supports two levels of reproducibility.

### 1. Frozen-output reproduction

Use the processed files in `data_public/` to reproduce the reported statistics, figures, and tables without re-running LLM annotation or PubMed retrieval.

This is the recommended route for exact manuscript reproduction because hosted LLM outputs may change over time.

### 2. Method rerun

Use the numbered scripts to re-run the pipeline on newly retrieved abstracts or your own abstracts. This requires local access to PubMed data, a configured OpenAI API key for annotation, and sentence-transformer models for embeddings.

## Public data policy

To reduce copyright risk, the public data files are redacted: full abstract text and sentence text are not redistributed. The repository includes PMIDs, metadata where available, sentence identifiers, rhetorical-role labels, confidence scores, and derived metrics. Users can rehydrate full abstracts locally from PubMed using the provided PMIDs and scripts.

## Corrected opening-frame convention

The current metric system uses the strict centroid-based opening-frame return:

`cosine(Conclusion centroid, Background/Aim opening-frame centroid)`

The older final-sentence-to-opening-anchor descriptor is treated as a superseded legacy diagnostic and is not used as evidence of added discriminative value.

## Citation

Please cite the manuscript and this repository. A `CITATION.cff` file is included and should be updated with the final DOI after publication.
