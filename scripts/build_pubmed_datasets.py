#!/usr/bin/env python
from pathlib import Path
import yaml
from rsm.pubmed import build_review_clinical_corpus

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

build_review_clinical_corpus(config, output_dir)
print("Done. PubMed candidate datasets saved in data/raw/")
