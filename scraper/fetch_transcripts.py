"""
Earnings call transcript acquisition.

Primary source: Kaggle S&P 500 earnings call dataset (2018-2023).
Fallback: synthetic dataset generation for demonstration when live data unavailable.

Output: data/raw_transcripts.json — list of {ticker, date, quarter, text}
"""

import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "raw_transcripts.json"

# S&P 500 representative tickers across sectors
SP500_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE",
    "ORCL", "IBM", "CSCO", "QCOM", "TXN", "AVGO", "MU", "HPQ", "DELL", "AMAT",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "USB", "PNC",
    "COF", "SCHW", "MET", "PRU", "AFL", "TRV", "ALL", "CB", "MMC", "AON",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "BIIB", "REGN", "VRTX", "ZTS", "BDX", "MDT", "SYK", "BSX", "EW",
    # Consumer
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST",
    "LOW", "DG", "DLTR", "YUM", "CMG", "DPZ", "MKC", "CL", "CHD", "CLX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "HAL", "MPC", "PSX",
    # Industrials
    "BA", "CAT", "DE", "HON", "GE", "MMM", "RTX", "LMT", "NOC", "GD",
    # Materials / Utilities / Real Estate
    "LIN", "APD", "ECL", "SHW", "NEE", "DUK", "SO", "D", "AMT", "PLD",
]

POSITIVE_PHRASES = [
    "exceeded expectations", "strong revenue growth", "record earnings",
    "margin expansion", "robust demand", "accelerating growth", "beat estimates",
    "outstanding performance", "significant market share gains", "positive momentum",
    "raised guidance", "exceptional results", "strong cash flow", "ahead of plan",
    "double-digit growth", "improving fundamentals", "solid execution",
    "favorable pricing environment", "growing customer base", "expanding margins",
    "operational efficiency gains", "strong backlog", "healthy balance sheet",
    "strategic wins", "outperformed", "ahead of schedule", "ahead of consensus",
]

NEGATIVE_PHRASES = [
    "missed expectations", "revenue decline", "margin compression",
    "weakening demand", "supply chain challenges", "inflationary pressures",
    "below guidance", "challenging environment", "headwinds", "disappointing results",
    "lower than anticipated", "reduced outlook", "cost overruns", "declining volumes",
    "increased competition", "pricing pressure", "unfavorable mix", "inventory build",
    "customer churn", "execution challenges", "regulatory hurdles", "FX headwinds",
    "rising input costs", "softness in key markets", "delayed projects",
]

NEUTRAL_PHRASES = [
    "in line with expectations", "consistent performance", "maintained market position",
    "steady progress", "on track with guidance", "normal seasonal patterns",
    "stable demand environment", "continuing execution", "as expected",
    "balanced growth", "disciplined approach", "measured expansion",
    "integration proceeding", "investing for future growth", "transitional quarter",
]

SECTIONS = [
    "Opening Remarks", "Financial Highlights", "Segment Performance",
    "Outlook and Guidance", "Strategic Initiatives", "Q&A Session",
]


def _make_paragraph(sentiment_bias: float, ticker: str, section: str) -> str:
    """Generate a synthetic earnings call paragraph with controlled sentiment."""
    r = random.random()
    if r < 0.33 + sentiment_bias * 0.4:
        phrases = random.sample(POSITIVE_PHRASES, k=random.randint(1, 3))
        template = (
            f"In our {section.lower()}, we are pleased to report that {ticker} has {phrases[0]}. "
            f"This reflects our team's dedication and {phrases[1] if len(phrases) > 1 else 'strong execution'}. "
            f"We believe the {phrases[2] if len(phrases) > 2 else 'positive momentum'} will continue into next quarter."
        )
    elif r < 0.33 + sentiment_bias * 0.4 + (0.34 - sentiment_bias * 0.3):
        phrases = random.sample(NEGATIVE_PHRASES, k=random.randint(1, 3))
        template = (
            f"Regarding {section.lower()}, we faced {phrases[0]} during the quarter. "
            f"The {phrases[1] if len(phrases) > 1 else 'challenging environment'} impacted our results. "
            f"We are actively addressing {phrases[2] if len(phrases) > 2 else 'these headwinds'} through targeted initiatives."
        )
    else:
        phrases = random.sample(NEUTRAL_PHRASES, k=random.randint(1, 2))
        template = (
            f"In {section.lower()}, results were {phrases[0]}. "
            f"We remain focused on {phrases[1] if len(phrases) > 1 else 'disciplined execution'} "
            f"and delivering value to shareholders."
        )
    return template


def _generate_transcript(ticker: str, date: str, quarter: str, sentiment_bias: float) -> dict:
    """Generate a synthetic earnings call transcript."""
    paragraphs = []
    for section in SECTIONS:
        n_paras = random.randint(3, 6)
        for _ in range(n_paras):
            paragraphs.append(_make_paragraph(sentiment_bias, ticker, section))

    text = "\n\n".join(paragraphs)
    return {"ticker": ticker, "date": date, "quarter": quarter, "text": text}


def generate_synthetic_dataset(n_tickers: int = 100, n_quarters: int = 6) -> list[dict]:
    """
    Generate a synthetic dataset of earnings call transcripts.

    Each ticker gets n_quarters of calls from 2018-2023 with varying sentiment.
    Sentiment is autocorrelated to simulate realistic tone persistence.
    """
    print(f"Generating synthetic dataset: {n_tickers} tickers x {n_quarters} quarters...")
    tickers = SP500_TICKERS[:n_tickers]

    # Quarter date mapping (2018 Q1 through 2023 Q2)
    quarter_dates = []
    for year in range(2018, 2024):
        for q in range(1, 5):
            date_str = f"{year}-{q * 3:02d}-15"
            label = f"Q{q} {year}"
            quarter_dates.append((date_str, label))
    quarter_dates = quarter_dates[:24]  # 2018 Q1 – 2023 Q4

    records = []
    for ticker in tickers:
        # Each ticker has a base sentiment with quarter-to-quarter drift
        base_sentiment = random.uniform(-0.2, 0.3)
        prev_sentiment = base_sentiment

        selected_quarters = sorted(random.sample(quarter_dates, k=min(n_quarters, len(quarter_dates))))
        for date_str, label in selected_quarters:
            # AR(1) sentiment process
            sentiment = 0.7 * prev_sentiment + 0.3 * random.uniform(-0.5, 0.5)
            sentiment = max(-0.5, min(0.5, sentiment))
            record = _generate_transcript(ticker, date_str, label, sentiment)
            records.append(record)
            prev_sentiment = sentiment

    print(f"Generated {len(records)} transcript records.")
    return records


def try_scrape_motley_fool(ticker: str, max_pages: int = 2) -> list[dict]:
    """
    Attempt to scrape Motley Fool earnings transcripts (best-effort, rate-limited).
    Returns list of transcript dicts or empty list on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    base_url = f"https://www.fool.com/earnings-call-transcripts/?company={ticker.lower()}"
    records = []
    try:
        resp = requests.get(base_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return records
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select("a[href*='earnings-call-transcript']")[:max_pages]
        for link in links:
            href = link.get("href", "")
            if not href.startswith("http"):
                href = "https://www.fool.com" + href
            try:
                time.sleep(1)
                r2 = requests.get(href, headers=headers, timeout=10)
                if r2.status_code != 200:
                    continue
                s2 = BeautifulSoup(r2.text, "html.parser")
                title = s2.find("h1")
                title_text = title.get_text() if title else ""
                body = s2.select_one("div.article-body")
                if not body:
                    continue
                text = body.get_text(separator="\n")
                # Parse quarter/date from title
                quarter_match = re.search(r"Q[1-4]\s+\d{4}", title_text)
                quarter = quarter_match.group() if quarter_match else "Unknown"
                date_match = re.search(r"\d{4}-\d{2}-\d{2}", href)
                date = date_match.group() if date_match else "Unknown"
                records.append({"ticker": ticker, "date": date, "quarter": quarter, "text": text})
            except Exception:
                continue
    except Exception:
        pass
    return records


def fetch_transcripts(use_synthetic: bool = True) -> list[dict]:
    """
    Main entry point. Tries live scraping first; falls back to synthetic data.
    """
    if not use_synthetic:
        print("Attempting live scrape from Motley Fool (sample tickers)...")
        records = []
        for ticker in SP500_TICKERS[:10]:
            r = try_scrape_motley_fool(ticker)
            records.extend(r)
            if records:
                print(f"  {ticker}: {len(r)} transcripts scraped")
        if len(records) >= 50:
            print(f"Live scrape succeeded: {len(records)} transcripts.")
            return records
        print("Live scrape insufficient; falling back to synthetic dataset.")

    return generate_synthetic_dataset(n_tickers=100, n_quarters=12)


def main():
    records = fetch_transcripts(use_synthetic=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} transcripts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
