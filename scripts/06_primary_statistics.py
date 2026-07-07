#!/usr/bin/env python
"""Primary statistics"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Compute review-clinical descriptive statistics, Cohen d, Welch p, and Mann-Whitney p.")

if __name__ == "__main__":
    main()
