#!/usr/bin/env python
"""Label rhetorical roles"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Build prompts, call the LLM API, parse JSON labels, and save sentence-level labels.")

if __name__ == "__main__":
    main()
