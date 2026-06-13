"""PubMed dataset builder for rhetorical-semantic coherence analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from Bio import Entrez, Medline
from tqdm import tqdm


def configure_entrez(email: str, api_key: Optional[str] = None) -> None:
    if not email or email == "your.email@example.com":
        raise ValueError("Please set a real NCBI email in config.yaml or via CLI.")
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def build_queries(topic_query: str, start_year: int, end_year: int) -> Dict[str, str]:
    date_filter = f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'

    review_query = f"""
    {topic_query}
    AND
    (
      Review[Publication Type]
      OR "systematic review"[tiab]
      OR "meta-analysis"[Publication Type]
      OR "meta analysis"[tiab]
    )
    NOT Case Reports[Publication Type]
    AND {date_filter}
    """

    clinical_query = f"""
    {topic_query}
    AND
    (
      Clinical Trial[Publication Type]
      OR Randomized Controlled Trial[Publication Type]
      OR observational study[tiab]
      OR cohort[tiab]
      OR cross-sectional[tiab]
      OR case-control[tiab]
      OR patients[tiab]
    )
    NOT
    (
      Review[Publication Type]
      OR systematic review[tiab]
      OR meta-analysis[Publication Type]
      OR meta analysis[tiab]
      OR Case Reports[Publication Type]
    )
    AND {date_filter}
    """

    return {"Review": review_query, "Clinical": clinical_query}


def search_pubmed_ids(query: str, retmax: int = 500, sleep_seconds: float = 0.35) -> List[str]:
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


def build_dataset(
    query: str,
    dataset_label: str,
    n: int = 50,
    retmax: int = 500,
    min_abstract_chars: int = 500,
    random_state: int = 42,
    sleep_seconds: float = 0.35,
    fetch_batch_size: int = 100,
) -> pd.DataFrame:
    pmids = search_pubmed_ids(query, retmax=retmax, sleep_seconds=sleep_seconds)
    print(f"{dataset_label}: found {len(pmids)} PMIDs")
    records = fetch_medline_records(pmids, batch_size=fetch_batch_size, sleep_seconds=sleep_seconds)
    df = pd.DataFrame([record_to_row(rec, dataset_label) for rec in records])
    if df.empty:
        return df
    df = df.drop_duplicates(subset="PMID")
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df = df[df["Abstract"].str.len() > min_abstract_chars].copy()
    df = df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)
    print(f"{dataset_label}: retained {len(df)} abstracts")
    return df


def build_review_clinical_corpus(config: dict, output_dir: Path) -> pd.DataFrame:
    pubmed_cfg = config["pubmed"]
    analysis_cfg = config.get("analysis", {})
    project_cfg = config.get("project", {})

    configure_entrez(pubmed_cfg["email"], pubmed_cfg.get("api_key"))
    queries = build_queries(pubmed_cfg["topic_query"], pubmed_cfg["start_year"], pubmed_cfg["end_year"])

    output_dir.mkdir(parents=True, exist_ok=True)

    df_review = build_dataset(
        queries["Review"],
        "Review",
        n=pubmed_cfg.get("n_per_group", 50),
        retmax=pubmed_cfg.get("retmax", 500),
        min_abstract_chars=analysis_cfg.get("min_abstract_chars", 500),
        random_state=project_cfg.get("random_state", 42),
        sleep_seconds=pubmed_cfg.get("sleep_seconds", 0.35),
        fetch_batch_size=pubmed_cfg.get("fetch_batch_size", 100),
    )

    df_clinical = build_dataset(
        queries["Clinical"],
        "Clinical",
        n=pubmed_cfg.get("n_per_group", 50),
        retmax=pubmed_cfg.get("retmax", 500),
        min_abstract_chars=analysis_cfg.get("min_abstract_chars", 500),
        random_state=project_cfg.get("random_state", 42),
        sleep_seconds=pubmed_cfg.get("sleep_seconds", 0.35),
        fetch_batch_size=pubmed_cfg.get("fetch_batch_size", 100),
    )

    overlap = set(df_review.get("PMID", [])).intersection(set(df_clinical.get("PMID", [])))
    if overlap:
        print(f"Removing {len(overlap)} overlapping PMIDs from Clinical dataset")
        df_clinical = df_clinical[~df_clinical["PMID"].isin(overlap)].copy()

    df_review.to_csv(output_dir / "periodontal_reviews_50.csv", index=False)
    df_clinical.to_csv(output_dir / "periodontal_clinical_50.csv", index=False)

    df_all = pd.concat([df_review, df_clinical], ignore_index=True).drop_duplicates(subset="PMID")
    df_all.to_csv(output_dir / "periodontal_reviews_vs_clinical_100.csv", index=False)
    return df_all
