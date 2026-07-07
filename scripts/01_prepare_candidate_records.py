#!/usr/bin/env python
"""Prepare candidate records"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Prepare or redact PubMed candidate records. Use candidate PMIDs/metadata as input.")

if __name__ == "__main__":
    main()
