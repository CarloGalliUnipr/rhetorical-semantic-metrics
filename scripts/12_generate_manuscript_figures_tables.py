#!/usr/bin/env python
"""Generate corrected manuscript figures and tables from frozen public outputs."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data_public" / "primary_statistics"
ABSTRACT = ROOT / "data_public" / "processed_abstract_level"
FIGURES = ROOT / "outputs" / "figures"

def generate_figure2():
    FIGURES.mkdir(parents=True, exist_ok=True)
    role = pd.read_csv(PRIMARY / "role_distribution_mean_sentence_counts.csv")
    review = pd.read_csv(ABSTRACT / "review_abstract_level_metrics.csv")
    clinical = pd.read_csv(ABSTRACT / "clinical_abstract_level_metrics.csv")

    role_order = ["Background", "Aim", "Methods", "Results", "Conclusion", "Limitation/Future"]
    metric_map = {
        "Mean local similarity": "mean_local_similarity",
        "Within-Results similarity": "within_Results_similarity",
        "Conclusion–Results alignment": "conclusion_to_results_alignment",
        "Opening-frame return": "conclusion_to_background_plus_aim_alignment",
    }

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(role_order))
    w = 0.36
    clinical_vals = [role.loc[role.Dataset == "Clinical", f"n_{r}"].iloc[0] for r in role_order]
    review_vals = [role.loc[role.Dataset == "Review", f"n_{r}"].iloc[0] for r in role_order]
    ax1.bar(x - w/2, clinical_vals, w, label="Clinical studies")
    ax1.bar(x + w/2, review_vals, w, label="Reviews/meta-analyses")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Background", "Aim", "Methods", "Results", "Conclusion", "Lim./Future"], rotation=15)
    ax1.set_ylabel("Mean number of sentences")
    ax1.set_title("A. Rhetorical-role composition", loc="left", fontweight="bold")
    ax1.legend(frameon=False, ncols=2, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x - w/2, clinical_vals):
        ax1.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w/2, review_vals):
        ax1.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    positions = np.arange(1, len(metric_map) + 1)
    offset = 0.18
    rng = np.random.default_rng(7)
    for i, (_, df) in enumerate([("Clinical studies", clinical), ("Reviews/meta-analyses", review)]):
        pos = positions + (-offset if i == 0 else offset)
        data = [df[col].dropna().values for col in metric_map.values()]
        bp = ax2.boxplot(data, positions=pos, widths=0.30, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_alpha(0.55)
        for p, vals in zip(pos, data):
            ax2.scatter(np.full(len(vals), p) + rng.normal(0, 0.04, len(vals)), vals, s=10, alpha=0.35)
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="gray")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(["Mean local\nsimilarity", "Within-Results\nsimilarity", "Conclusion–Results\nalignment", "Opening-frame\nreturn"])
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("B. Generic local continuity versus role-aware metrics", loc="left", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    effect_data = pd.DataFrame([
        ("Conclusion–Results alignment", 0.644),
        ("Opening-frame return", 0.201),
        ("Mean local similarity", -0.171),
        ("Within-Results similarity", -0.705),
    ], columns=["Metric", "d"]).iloc[::-1]
    ax3.barh(effect_data["Metric"], effect_data["d"])
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_xlim(-0.9, 0.9)
    ax3.set_xlabel("Cohen's d (review minus clinical study)")
    ax3.set_title("C. Direction and magnitude of review–clinical differences", loc="left", fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)
    for y, val in enumerate(effect_data["d"]):
        ax3.text(val + (0.03 if val >= 0 else -0.03), y, f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    ax3.text(-0.85, -0.75, "Clinical higher", ha="left", fontsize=9)
    ax3.text(0.85, -0.75, "Reviews higher", ha="right", fontsize=9)

    fig.suptitle("Figure 2. Primary demonstration analyses of the rhetorical-semantic framework", fontsize=14, fontweight="bold")
    fig.savefig(FIGURES / "figure2_corrected.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figure2_corrected.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    generate_figure2()
    print("Generated corrected Figure 2 in outputs/figures.")

if __name__ == "__main__":
    main()
