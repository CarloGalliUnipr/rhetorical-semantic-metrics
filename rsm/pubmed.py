"""PubMed dataset builder for rhetorical-semantic metric analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from Bio import Entrez, Medline
from tqdm import tqdm


def configure_entrez(email: str, api_key: Optional[str] = None) -> None:
    if not email or email == "your.email@example.com":
        raise ValueError("Please set a real NCBI email in config.yaml.")
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def build_queries(topic_query: str, start_year: int, end_year: int) -> Dict[str, str]:
    """Build article-type-specific PubMed queries.

    Review candidates are restricted to systematic reviews/meta-analyses to ensure
    identifiable Methods and Results components. Narrative reviews are intentionally
    excluded from the default pilot design.
    """
    date_filter = f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'

    review_query = f"""
    {topic_query}
    AND
    (
      Systematic Review[Publication Type]
      OR Meta-Analysis[Publication Type]
      OR "systematic review"[tiab]
      OR "meta-analysis"[tiab]
      OR "meta analysis"[tiab]
    )
    NOT
    (
      Case Reports[Publication Type]
      OR Editorial[Publication Type]
      OR Letter[Publication Type]
      OR Comment[Publication Type]
    )
    AND {date_filter}
    """

    clinical_query = f"""
    {topic_query}
    AND
    (
      Clinical Trial[Publication Type]
      OR Randomized Controlled Trial[Publication Type]
      OR Controlled Clinical Trial[Publication Type]
      OR observational study[tiab]
      OR cohort[tiab]
      OR cross-sectional[tiab]
      OR case-control[tiab]
      OR patients[tiab]
    )
    NOT
    (
      Review[Publication Type]
      OR Systematic Review[Publication Type]
      OR Meta-Analysis[Publication Type]
      OR "systematic review"[tiab]
      OR "meta-analysis"[tiab]
      OR "meta analysis"[tiab]
      OR Case Reports[Publication Type]
      OR Editorial[Publication Type]
      OR Letter[Publication Type]
      OR Comment[Publication Type]
    )
    AND {date_filter}
    """

    return {"Review": review_query, "Clinical": clinical_query}


def search_pubmed_ids(query: str, retmax: int = 1000, sleep_seconds: float = 0.35) -> List[str]:
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=retmax,
        retmode="xml",
        sort="relevance",
    )
    results = Entrez.read(handle)
    handle.close()
    time.sleep(sleep_seconds)
    return list(results.get("IdList", []))


def fetch_medline_records(pmids: List[str], batch_size: int = 100, sleep_seconds: float = 0.35) -> List[dict]:
    records: List[dict] = []
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching MEDLINE"):
        batch = pmids[i : i + batch_size]
        if not batch:
            continue
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="medline",
            retmode="text",
        )
        records.extend(list(Medline.parse(handle)))
        handle.close()
        time.sleep(sleep_seconds)
    return records


def record_to_row(record: dict, dataset_label: str) -> dict:
    return {
        "PMID": record.get("PMID", ""),
        "Title": record.get("TI", ""),
        "Abstract": record.get("AB", ""),
        "Year": record.get("DP", ""),
        "Journal": record.get("JT", ""),
        "Authors": "; ".join(record.get("AU", [])) if isinstance(record.get("AU", []), list) else "",
        "PublicationTypes": "; ".join(record.get("PT", [])) if isinstance(record.get("PT", []), list) else "",
        "Mesh": "; ".join(record.get("MH", [])) if isinstance(record.get("MH", []), list) else "",
        "Dataset": dataset_label,
    }


def build_candidate_dataset(
    query: str,
    dataset_label: str,
    candidates_n: int = 200,
    retmax: int = 1000,
    random_state: int = 42,
    sleep_seconds: float = 0.35,
    fetch_batch_size: int = 100,
) -> pd.DataFrame:
    """Build a candidate dataset without hard filtering by abstract length."""
    pmids = search_pubmed_ids(query, retmax=retmax, sleep_seconds=sleep_seconds)
    print(f"{dataset_label}: found {len(pmids)} PMIDs")
    records = fetch_medline_records(pmids, batch_size=fetch_batch_size, sleep_seconds=sleep_seconds)
    df = pd.DataFrame([record_to_row(rec, dataset_label) for rec in records])
    if df.empty:
        return df

    df = df.drop_duplicates(subset="PMID")
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Title"] = df["Title"].fillna("").astype(str)
    df["abstract_length_chars"] = df["Abstract"].str.len()
    df["abstract_word_count"] = df["Abstract"].str.split().str.len()

    # Relevance sorting comes from PubMed; sample with seed only if more candidates are retrieved.
    if len(df) > candidates_n:
        df = df.sample(n=candidates_n, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"{dataset_label}: retained {len(df)} candidate abstracts")
    return df


def build_review_clinical_corpus(config: dict, output_dir: Path) -> pd.DataFrame:
    pubmed_cfg = config["pubmed"]
    project_cfg = config.get("project", {})

    configure_entrez(pubmed_cfg["email"], pubmed_cfg.get("api_key"))
    queries = build_queries(pubmed_cfg["topic_query"], pubmed_cfg["start_year"], pubmed_cfg["end_year"])

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_n = pubmed_cfg.get("candidates_per_group", 200)

    df_review = build_candidate_dataset(
        queries["Review"],
        "Review",
        candidates_n=candidates_n,
        retmax=pubmed_cfg.get("retmax", 1000),
        random_state=project_cfg.get("random_state", 42),
        sleep_seconds=pubmed_cfg.get("sleep_seconds", 0.35),
        fetch_batch_size=pubmed_cfg.get("fetch_batch_size", 100),
    )

    df_clinical = build_candidate_dataset(
        queries["Clinical"],
        "Clinical",
        candidates_n=candidates_n,
        retmax=pubmed_cfg.get("retmax", 1000),
        random_state=project_cfg.get("random_state", 42),
        sleep_seconds=pubmed_cfg.get("sleep_seconds", 0.35),
        fetch_batch_size=pubmed_cfg.get("fetch_batch_size", 100),
    )

    overlap = set(df_review.get("PMID", [])).intersection(set(df_clinical.get("PMID", [])))
    if overlap:
        print(f"Removing {len(overlap)} overlapping PMIDs from Clinical candidates")
        df_clinical = df_clinical[~df_clinical["PMID"].isin(overlap)].copy()

    review_path = output_dir / f"periodontal_reviews_{len(df_review)}_candidates.csv"
    clinical_path = output_dir / f"periodontal_clinical_{len(df_clinical)}_candidates.csv"

    df_review.to_csv(review_path, index=False)
    df_clinical.to_csv(clinical_path, index=False)

    df_all = pd.concat([df_review, df_clinical], ignore_index=True).drop_duplicates(subset="PMID")
    df_all.to_csv(output_dir / "periodontal_reviews_vs_clinical_candidates.csv", index=False)

    print(f"Saved review candidates to {review_path}")
    print(f"Saved clinical candidates to {clinical_path}")
    return df_all
