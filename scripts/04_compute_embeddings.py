#!/usr/bin/env python
"""Compute embeddings"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Compute sentence embeddings with all-MiniLM-L6-v2 or another sentence-transformer model.")

if __name__ == "__main__":
    main()
