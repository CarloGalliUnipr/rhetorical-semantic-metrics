"""Role-aware rhetorical-semantic metrics for scientific abstracts."""

from .metrics import (
    cosine,
    normalize_vector,
    role_centroids,
    mean_local_similarity,
    within_role_similarity,
    cross_role_alignment,
    opening_frame_return,
    compute_abstract_metrics,
)
from .statistics import cohens_d, compare_groups
