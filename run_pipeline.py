"""
End-to-end pipeline runner.

Usage:
  python run_pipeline.py              # full pipeline, mock FinBERT
  python run_pipeline.py --real       # use real FinBERT (GPU recommended)
  python run_pipeline.py --skip-scrape  # skip scraping if data already exists
"""

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real FinBERT (not mock)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping step")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    # Step 1: Acquire transcripts
    if not args.skip_scrape or not (DATA_DIR / "raw_transcripts.json").exists():
        step("1 / 4 — Transcript Acquisition")
        from scraper.fetch_transcripts import main as scrape_main
        scrape_main()
    else:
        print("Skipping scrape (raw_transcripts.json exists).")

    # Step 2: FinBERT scoring
    step("2 / 4 — FinBERT NLP Scoring")
    from nlp.finbert_scorer import main as nlp_main
    nlp_main(use_mock=not args.real)

    # Step 3: IC analysis
    step("3 / 4 — Signal IC Analysis")
    from signals.ic_analysis import main as signal_main
    signal_main()

    # Step 4: Backtest
    step("4 / 4 — Long-Short Backtest")
    from backtest.longshort_backtest import main as backtest_main
    backtest_main()

    print("\n" + "="*60)
    print("  Pipeline complete. Check outputs/ for results.")
    print("="*60)


if __name__ == "__main__":
    main()
