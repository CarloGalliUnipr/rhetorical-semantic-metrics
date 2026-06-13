"""Batch analysis, tables, and plots for rhetorical-semantic metrics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import mannwhitneyu, ttest_ind
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .labeling import RHETORICAL_ROLES
from .metrics import analyze_single_abstract, get_spacy_model

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

SELECTED_PLOT_METRICS = [
    "mean_local_similarity",
    "local_similarity_volatility",
    "within_Methods_similarity",
    "within_Results_similarity",
    "conclusion_to_aim_alignment",
    "conclusion_to_results_alignment",
    "results_to_methods_alignment",
]


def analyze_dataset(
    df_input: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    nlp,
    embedding_model: SentenceTransformer,
    openai_client: OpenAI,
    llm_model_name: str,
    min_sentences: int = 5,
    sleep_between_llm_calls: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sentence_level_rows = []
    summary_rows = []
    error_rows = []

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"Analyzing {dataset_name}"):
        pmid = row.get("PMID", None)
        title = row.get("Title", None)
        abstract = row.get("Abstract", "")
        try:
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
                sleep_between_llm_calls=sleep_between_llm_calls,
                verbose=False,
            )
            sentence_level_rows.append(df_sent)
            summary_rows.append(summary)
        except Exception as e:
            error_rows.append({"PMID": pmid, "Title": title, "Dataset": dataset_name, "Error": str(e)})
            print(f"Error in PMID {pmid}: {e}")

    sentence_df = pd.concat(sentence_level_rows, ignore_index=True) if sentence_level_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    error_df = pd.DataFrame(error_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    sentence_df.to_csv(output_dir / f"{dataset_name.lower()}_sentence_level_metrics.csv", index=False)
    summary_df.to_csv(output_dir / f"{dataset_name.lower()}_abstract_level_metrics.csv", index=False)
    error_df.to_csv(output_dir / f"{dataset_name.lower()}_errors.csv", index=False)
    print(f"{dataset_name}: completed {len(summary_df)} abstracts; errors: {len(error_df)}")
    return sentence_df, summary_df, error_df


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

    role_count_cols = [f"n_{role}" for role in RHETORICAL_ROLES]
    role_distribution = summary_all.groupby("Dataset")[role_count_cols].mean()
    role_distribution.to_csv(tables_dir / "rhetorical_role_distribution.csv")

    confidence_by_role = sentence_all.groupby(["Dataset", "rhetorical_role"])["label_confidence"].agg(["mean", "std", "count"]).reset_index()
    confidence_by_role.to_csv(tables_dir / "llm_confidence_by_role_and_dataset.csv", index=False)

    role_metric_summary = sentence_all.groupby(["Dataset", "rhetorical_role"]).agg(
        mean_local_similarity=("local_similarity_to_previous_sentence", "mean"),
        mean_similarity_to_opening=("similarity_to_opening_anchor", "mean"),
        mean_own_role_similarity_loo=("similarity_to_own_role_anchor_loo", "mean"),
        n_sentences=("sentence", "count"),
    ).reset_index()
    role_metric_summary.to_csv(tables_dir / "role_level_metric_summary.csv", index=False)


def make_plots(summary_all: pd.DataFrame, sentence_all: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected_metrics = [m for m in SELECTED_PLOT_METRICS if m in summary_all.columns]
    plot_df = summary_all.groupby("Dataset")[selected_metrics].agg(["mean", "sem"])

    x = np.arange(len(selected_metrics))
    width = 0.35
    review_means = [plot_df.loc["Review", (m, "mean")] for m in selected_metrics]
    clinical_means = [plot_df.loc["Clinical", (m, "mean")] for m in selected_metrics]
    review_sems = [plot_df.loc["Review", (m, "sem")] for m in selected_metrics]
    clinical_sems = [plot_df.loc["Clinical", (m, "sem")] for m in selected_metrics]

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, review_means, width, yerr=review_sems, capsize=4, label="Reviews")
    plt.bar(x + width / 2, clinical_means, width, yerr=clinical_sems, capsize=4, label="Clinical studies")
    plt.xticks(x, selected_metrics, rotation=45, ha="right")
    plt.ylabel("Mean cosine similarity")
    plt.title("Rhetorical-semantic metrics: Reviews vs Clinical studies")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "barplot_reviews_vs_clinical_core_metrics.png", dpi=300)
    plt.close()

    for metric in selected_metrics:
        data_review = summary_all.loc[summary_all["Dataset"] == "Review", metric].dropna()
        data_clinical = summary_all.loc[summary_all["Dataset"] == "Clinical", metric].dropna()
        plt.figure(figsize=(6, 5))
        plt.boxplot([data_review, data_clinical], labels=["Reviews", "Clinical"])
        plt.ylabel(metric)
        plt.title(f"{metric}: Reviews vs Clinical studies")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"boxplot_{metric}.png", dpi=300)
        plt.close()

    role_count_cols = [f"n_{role}" for role in RHETORICAL_ROLES]
    role_distribution = summary_all.groupby("Dataset")[role_count_cols].mean()
    role_distribution.T.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Mean number of sentences")
    plt.title("Mean rhetorical-role composition by dataset")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "rhetorical_role_distribution_reviews_vs_clinical.png", dpi=300)
    plt.close()

    confidence_by_role = sentence_all.groupby(["Dataset", "rhetorical_role"])["label_confidence"].agg(["mean"]).reset_index()
    confidence_pivot = confidence_by_role.pivot(index="rhetorical_role", columns="Dataset", values="mean").reindex(RHETORICAL_ROLES)
    confidence_pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Mean LLM confidence")
    plt.title("LLM rhetorical-label confidence by role")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "llm_confidence_by_role_reviews_vs_clinical.png", dpi=300)
    plt.close()


def run_full_analysis(config: dict, input_csv: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_all = pd.read_csv(input_csv)
    df_all["Abstract"] = df_all["Abstract"].fillna("").astype(str)
    df_all["Title"] = df_all["Title"].fillna("").astype(str)

    nlp = get_spacy_model()
    embedding_model = SentenceTransformer(config["models"]["embedding_model"])
    openai_client = OpenAI()

    sentence_outputs = []
    summary_outputs = []
    for dataset_name in ["Review", "Clinical"]:
        df_group = df_all[df_all["Dataset"] == dataset_name].copy()
        sent_df, summ_df, _ = analyze_dataset(
            df_group,
            dataset_name,
            output_dir,
            nlp,
            embedding_model,
            openai_client,
            config["models"]["llm_model"],
            min_sentences=config.get("analysis", {}).get("min_sentences", 5),
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
