"""Batch analysis, tables, and plots for rhetorical-semantic metrics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import mannwhitneyu, ttest_ind
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .labeling import RHETORICAL_ROLES
from .metrics import analyze_single_abstract, get_spacy_model, split_into_sentences

CORE_METRICS = [
    "n_sentences",
    "mean_label_confidence",
    "mean_local_similarity",
    "min_local_similarity",
    "local_similarity_volatility",
    "local_discontinuity_burden",
    "within_role_discontinuity_burden",
    "mean_similarity_to_opening_anchor",
    "final_similarity_to_opening_anchor",
    "mean_similarity_to_own_role_anchor_loo",
    "within_Methods_similarity",
    "within_Results_similarity",
    "conclusion_to_aim_alignment",
    "conclusion_to_results_alignment",
    "results_to_methods_alignment",
    "methods_to_aim_alignment",
    "results_to_aim_alignment",
    "conclusion_to_background_plus_aim_alignment",
    "conclusion_plus_future_to_background_plus_aim_alignment",
]

FIGURE2_METRICS = [
    "mean_local_similarity",
    "within_Results_similarity",
    "conclusion_to_results_alignment",
]

ERROR_COLUMNS = ["PMID", "Title", "Dataset", "Error", "Abstract"]
EXCLUSION_COLUMNS = ["PMID", "Title", "Dataset", "Reason", "n_sentences_detected", "Abstract"]


def _empty_dataframe(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def analyze_dataset_until_n_success(
    df_input: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    nlp,
    embedding_model: SentenceTransformer,
    openai_client: OpenAI,
    llm_model_name: str,
    target_n: int = 50,
    min_sentences: int = 5,
    required_roles: Sequence[str] = ("Methods", "Results"),
    sleep_between_llm_calls: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze candidates until a target number of successful abstracts is reached."""
    sentence_level_rows = []
    summary_rows = []
    error_rows = []
    excluded_rows = []

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"Analyzing {dataset_name}"):
        if len(summary_rows) >= target_n:
            break

        pmid = row.get("PMID", None)
        title = row.get("Title", None)
        abstract = str(row.get("Abstract", ""))

        try:
            sentence_preview = split_into_sentences(abstract, nlp)
            if len(sentence_preview) < min_sentences:
                excluded_rows.append({
                    "PMID": pmid,
                    "Title": title,
                    "Dataset": dataset_name,
                    "Reason": f"Fewer than {min_sentences} sentences",
                    "n_sentences_detected": len(sentence_preview),
                    "Abstract": abstract,
                })
                continue

            df_sent, summary = analyze_single_abstract(
                abstract_text=abstract,
                pmid=pmid,
                title=title,
                dataset=dataset_name,
                nlp=nlp,
                embedding_model=embedding_model,
                openai_client=openai_client,
                llm_model_name=llm_model_name,
                min_sentences=min_sentences,
                required_roles=required_roles,
                sleep_between_llm_calls=sleep_between_llm_calls,
                verbose=False,
            )

            sentence_level_rows.append(df_sent)
            summary_rows.append(summary)

        except Exception as e:
            error_rows.append({
                "PMID": pmid,
                "Title": title,
                "Dataset": dataset_name,
                "Error": str(e),
                "Abstract": abstract,
            })
            print(f"Error/exclusion in PMID {pmid}: {e}")

    sentence_df = pd.concat(sentence_level_rows, ignore_index=True) if sentence_level_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    error_df = pd.DataFrame(error_rows) if error_rows else _empty_dataframe(ERROR_COLUMNS)
    excluded_df = pd.DataFrame(excluded_rows) if excluded_rows else _empty_dataframe(EXCLUSION_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    sentence_df.to_csv(output_dir / f"{dataset_name.lower()}_sentence_level_metrics.csv", index=False)
    summary_df.to_csv(output_dir / f"{dataset_name.lower()}_abstract_level_metrics.csv", index=False)
    error_df.to_csv(output_dir / f"{dataset_name.lower()}_errors.csv", index=False)
    excluded_df.to_csv(output_dir / f"{dataset_name.lower()}_excluded_before_llm.csv", index=False)

    print(
        f"{dataset_name}: completed {len(summary_df)} abstracts; "
        f"errors: {len(error_df)}; pre-LLM exclusions: {len(excluded_df)}"
    )
    if len(summary_df) < target_n:
        print(f"WARNING: target_n={target_n}, but only {len(summary_df)} successful abstracts were obtained.")

    return sentence_df, summary_df, error_df, excluded_df


def mean_sd_table(df: pd.DataFrame, metrics: List[str], group_col: str = "Dataset") -> pd.DataFrame:
    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for group in sorted(df[group_col].dropna().unique()):
            values = df.loc[df[group_col] == group, metric].dropna()
            row[group] = "NA" if len(values) == 0 else f"{values.mean():.3f} ± {values.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = np.array(x.dropna())
    y = np.array(y.dropna())
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return np.nan if pooled_sd == 0 else (np.mean(x) - np.mean(y)) / pooled_sd


def statistical_comparison(summary_all: pd.DataFrame, metrics: List[str] = CORE_METRICS) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in summary_all.columns:
            continue
        review_values = summary_all.loc[summary_all["Dataset"] == "Review", metric].dropna()
        clinical_values = summary_all.loc[summary_all["Dataset"] == "Clinical", metric].dropna()
        if len(review_values) >= 3 and len(clinical_values) >= 3:
            _, t_p = ttest_ind(review_values, clinical_values, equal_var=False, nan_policy="omit")
            _, u_p = mannwhitneyu(review_values, clinical_values, alternative="two-sided")
            rows.append({
                "Metric": metric,
                "Review_mean": review_values.mean(),
                "Clinical_mean": clinical_values.mean(),
                "Review_SD": review_values.std(),
                "Clinical_SD": clinical_values.std(),
                "Mean_difference_Review_minus_Clinical": review_values.mean() - clinical_values.mean(),
                "Cohens_d": cohens_d(review_values, clinical_values),
                "Welch_t_p": t_p,
                "Mann_Whitney_p": u_p,
                "n_review": len(review_values),
                "n_clinical": len(clinical_values),
            })
    return pd.DataFrame(rows)


def make_tables(summary_all: pd.DataFrame, sentence_all: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_summary = summary_all.groupby("Dataset")[CORE_METRICS].agg(["mean", "std", "sem", "count"])
    metric_summary.to_csv(tables_dir / "dataset_level_metric_summary_mean_sd_sem.csv")

    mean_sd = mean_sd_table(summary_all, CORE_METRICS)
    mean_sd.to_csv(tables_dir / "table_mean_sd_reviews_vs_clinical.csv", index=False)

    comparison = statistical_comparison(summary_all, CORE_METRICS)
    comparison.to_csv(tables_dir / "statistical_comparison_reviews_vs_clinical.csv", index=False)
    comparison.sort_values("Cohens_d", key=lambda s: s.abs(), ascending=False).to_csv(
        tables_dir / "statistical_comparison_sorted_by_effect_size.csv",
        index=False,
    )

    role_count_cols = [f"n_{role}" for role in RHETORICAL_ROLES]
    role_distribution = summary_all.groupby("Dataset")[role_count_cols].mean()
    role_distribution.to_csv(tables_dir / "rhetorical_role_distribution.csv")

    confidence_by_role = (
        sentence_all.groupby(["Dataset", "rhetorical_role"])["label_confidence"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    confidence_by_role.to_csv(tables_dir / "llm_confidence_by_role_and_dataset.csv", index=False)

    role_metric_summary = sentence_all.groupby(["Dataset", "rhetorical_role"]).agg(
        mean_local_similarity=("local_similarity_to_previous_sentence", "mean"),
        mean_similarity_to_opening=("similarity_to_opening_anchor", "mean"),
        mean_own_role_similarity_loo=("similarity_to_own_role_anchor_loo", "mean"),
        n_sentences=("sentence", "count"),
    ).reset_index()
    role_metric_summary.to_csv(tables_dir / "role_level_metric_summary.csv", index=False)

    file_manifest = []
    for p in output_dir.rglob("*"):
        if p.is_file():
            file_manifest.append({
                "filename": p.name,
                "relative_path": str(p.relative_to(output_dir)),
                "size_bytes": p.stat().st_size,
            })
    pd.DataFrame(file_manifest).sort_values("relative_path").to_csv(tables_dir / "generated_file_manifest.csv", index=False)


def make_plots(summary_all: pd.DataFrame, sentence_all: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Main Figure 2 source plot: local vs role-aware metrics.
    selected = [m for m in FIGURE2_METRICS if m in summary_all.columns]
    fig, axes = plt.subplots(1, len(selected), figsize=(13.5, 5.2), sharey=False)
    if len(selected) == 1:
        axes = [axes]

    titles = {
        "mean_local_similarity": "A. Local continuity",
        "within_Results_similarity": "B. Within-Results similarity",
        "conclusion_to_results_alignment": "C. Conclusion–Results alignment",
    }
    ylabels = {
        "mean_local_similarity": "Mean adjacent-sentence similarity",
        "within_Results_similarity": "Within-Results similarity",
        "conclusion_to_results_alignment": "Conclusion–Results alignment",
    }

    rng = np.random.default_rng(42)
    for i, metric in enumerate(selected):
        ax = axes[i]
        data_review = summary_all.loc[summary_all["Dataset"] == "Review", metric].dropna()
        data_clinical = summary_all.loc[summary_all["Dataset"] == "Clinical", metric].dropna()
        data = [data_review, data_clinical]
        box = ax.boxplot(
            data,
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": 1.6, "color": "black"},
            boxprops={"linewidth": 1.2, "edgecolor": "black"},
            whiskerprops={"linewidth": 1.2, "color": "black"},
            capprops={"linewidth": 1.2, "color": "black"},
        )
        box["boxes"][0].set_facecolor("white")
        box["boxes"][1].set_facecolor("0.85")
        for j, values in enumerate(data, start=1):
            ax.scatter(rng.normal(j, 0.045, size=len(values)), values, s=18, alpha=0.55, color="black", edgecolors="none")
        ax.scatter([1, 2], [data_review.mean(), data_clinical.mean()], marker="D", s=45, color="black", zorder=4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Reviews", "Clinical\nstudies"])
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabels.get(metric, metric))
        ax.set_title(titles.get(metric, metric), fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Local and role-aware rhetorical-semantic metrics in the pilot corpus", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure2_local_vs_role_aware_metrics.png", dpi=600, bbox_inches="tight")
    plt.close()

    # Supplementary Figure S1: role composition.
    role_count_cols = [f"n_{role}" for role in RHETORICAL_ROLES]
    role_distribution = summary_all.groupby("Dataset")[role_count_cols].agg(["mean", "sem"])
    x = np.arange(len(RHETORICAL_ROLES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))
    review_means = [role_distribution.loc["Review", (f"n_{role}", "mean")] for role in RHETORICAL_ROLES]
    clinical_means = [role_distribution.loc["Clinical", (f"n_{role}", "mean")] for role in RHETORICAL_ROLES]
    review_sem = [role_distribution.loc["Review", (f"n_{role}", "sem")] for role in RHETORICAL_ROLES]
    clinical_sem = [role_distribution.loc["Clinical", (f"n_{role}", "sem")] for role in RHETORICAL_ROLES]
    ax.bar(x - width / 2, review_means, width, yerr=review_sem, capsize=4, label="Reviews", edgecolor="black", facecolor="white")
    ax.bar(x + width / 2, clinical_means, width, yerr=clinical_sem, capsize=4, label="Clinical studies", edgecolor="black", facecolor="0.85")
    ax.set_xticks(x)
    ax.set_xticklabels(RHETORICAL_ROLES, rotation=35, ha="right")
    ax.set_ylabel("Mean number of sentences")
    ax.set_title("Rhetorical-role composition of abstracts", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "figureS1_rhetorical_role_composition.png", dpi=600, bbox_inches="tight")
    plt.close()


def run_full_analysis(config: dict, input_csv: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_all = pd.read_csv(input_csv)
    df_all["Abstract"] = df_all["Abstract"].fillna("").astype(str)
    df_all["Title"] = df_all["Title"].fillna("").astype(str)

    nlp = get_spacy_model()
    embedding_model = SentenceTransformer(config["models"]["embedding_model"])
    openai_client = OpenAI()

    target_n = config.get("pubmed", {}).get("target_successful_per_group", 50)
    min_sentences = config.get("analysis", {}).get("min_sentences", 5)
    required_roles = config.get("analysis", {}).get("required_roles", ["Methods", "Results"])

    sentence_outputs = []
    summary_outputs = []
    for dataset_name in ["Review", "Clinical"]:
        df_group = df_all[df_all["Dataset"] == dataset_name].copy()
        sent_df, summ_df, _, _ = analyze_dataset_until_n_success(
            df_group,
            dataset_name,
            output_dir,
            nlp,
            embedding_model,
            openai_client,
            config["models"]["llm_model"],
            target_n=target_n,
            min_sentences=min_sentences,
            required_roles=required_roles,
            sleep_between_llm_calls=config["models"].get("sleep_between_llm_calls", 0.5),
        )
        sentence_outputs.append(sent_df)
        summary_outputs.append(summ_df)

    sentence_all = pd.concat(sentence_outputs, ignore_index=True)
    summary_all = pd.concat(summary_outputs, ignore_index=True)
    sentence_all.to_csv(output_dir / "all_sentence_level_metrics.csv", index=False)
    summary_all.to_csv(output_dir / "all_abstract_level_metrics.csv", index=False)
    make_tables(summary_all, sentence_all, output_dir)
    make_plots(summary_all, sentence_all, output_dir)
    return sentence_all, summary_all
