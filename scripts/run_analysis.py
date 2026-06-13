#!/usr/bin/env python
from pathlib import Path
import yaml
from rsm.analysis import run_full_analysis

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

input_csv = Path("data/raw/periodontal_reviews_vs_clinical_100.csv")
output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
output_dir.mkdir(parents=True, exist_ok=True)

run_full_analysis(config, input_csv, output_dir)
print("Done. Analysis outputs saved in outputs/")
