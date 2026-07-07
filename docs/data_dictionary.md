# Data dictionary

Common fields:

- `PMID`: PubMed identifier.
- `Dataset`: abstract type, usually `Review` or `Clinical`.
- `Title`: article title when available.
- `sentence_id`: 1-based sentence index within an abstract.
- `label`: rhetorical-role label assigned to a sentence.
- `confidence`: LLM-reported confidence score.
- `n_sentences`: number of retained sentence-level units in the abstract.
- `mean_label_confidence`: mean LLM confidence across sentences in the abstract.
- `mean_local_similarity`: mean adjacent sentence-to-sentence cosine similarity.
- `within_Results_similarity`: mean pairwise cosine similarity among Results sentences.
- `conclusion_to_results_alignment`: cosine similarity between Conclusion and Results centroids.
- `results_to_methods_alignment`: cosine similarity between Results and Methods centroids.
- `methods_to_aim_alignment`: exploratory cosine similarity between Methods and Aim centroids.
- `results_to_aim_alignment`: exploratory cosine similarity between Results and Aim centroids.
- `conclusion_to_background_plus_aim_alignment`: strict opening-frame return.
- `conclusion_plus_future_to_background_plus_aim_alignment`: broader final-interpretive opening-frame sensitivity metric.

Public sentence-level files are redacted and do not include sentence text.
