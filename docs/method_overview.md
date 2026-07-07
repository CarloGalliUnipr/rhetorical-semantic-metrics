# Method overview

The workflow follows these steps:

1. Prepare PubMed candidate records.
2. Segment abstracts into ordered sentences.
3. Classify each sentence into a rhetorical role.
4. Compute sentence embeddings.
5. Compute abstract-level rhetorical-semantic metrics.
6. Compare review/meta-analysis and clinical-study abstracts.
7. Run boundary analyses: sentence-count sensitivity, embedding-model comparison, positional-baseline ablation, and surface/lexical baselines.
8. Run diagnostic perturbation experiments.
9. Generate manuscript figures and tables.

The unit of analysis is the abstract. Sentence-level outputs are intermediate records used to derive abstract-level metrics.

The review-clinical contrast is a validation setting for metric behavior, not a substantive claim about periodontal genres.
