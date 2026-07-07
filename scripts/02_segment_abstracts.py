#!/usr/bin/env python
"""Segment abstracts"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Segment full abstracts into ordered sentence-level units. Requires local full abstract text.")

if __name__ == "__main__":
    main()
