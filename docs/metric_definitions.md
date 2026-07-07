# Metric definitions

## Local continuity

For sentence `i > 1`, local similarity is:

`L_i = cos(e_i, e_{i-1})`

The abstract-level mean local similarity is the mean of `L_i` across adjacent sentence pairs.

## Within-role consistency

For a role containing at least two sentences, within-role consistency is the mean pairwise cosine similarity among all sentences assigned to that role.

The primary within-role metric in the manuscript is:

`within_Results_similarity`

## Cross-role alignment

For two roles, cross-role alignment is the cosine similarity between their role-specific centroids.

The primary cross-role metric is:

`conclusion_to_results_alignment`

## Opening-frame return

The opening frame is the centroid of Background and Aim sentences.

The strict opening-frame return metric is:

`cos(Conclusion centroid, Background/Aim opening-frame centroid)`

This is the current Methods-defined opening-frame metric. The older final-sentence-to-opening-anchor descriptor is retained only as a legacy diagnostic if present in machine-readable outputs.
