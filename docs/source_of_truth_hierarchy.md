# Source-of-truth hierarchy

The project passed through several analysis stages. For exact reproduction of the current manuscript, use the following hierarchy.

## Primary MiniLM analysis

Use `data_public/supplementary_outputs/frozen_source_zips/rhetorical_semantic_analysis_csv.zip`.

Key tables are extracted into `data_public/primary_statistics/`.

## Embedding-model comparison

Use `data_public/supplementary_outputs/frozen_source_zips/embedding_model_comparison_csv.zip`.

This analysis supports directional stability across MiniLM and MPNet only. It does not establish full embedding independence.

## Sentence-count sensitivity

Use `data_public/supplementary_outputs/frozen_source_zips/sentence_count_confound_analysis_csv.zip`.

The matched-pair analysis is supportive but unstable because only seven pairs could be matched. The adjusted and residualized analyses carry the sentence-count argument.

## Surface and lexical controls

Use `data_public/boundary_analyses/surface_lexical/`.

## Perturbation diagnostics

Use `data_public/supplementary_outputs/frozen_source_zips/perturbation_experiments_csv.zip`.

These experiments characterize metric behavior. They are not independent validation of rhetorical labels or genre separation.

## Opening-frame return

For opening-frame return, use the corrected symmetric centroid files in `data_public/boundary_analyses/opening_frame_corrected/`.

Do not use the old final-sentence-to-opening-anchor descriptor as the primary opening-frame metric.
