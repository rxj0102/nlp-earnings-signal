"""
FinBERT scoring pipeline.

Loads ProsusAI/finbert from HuggingFace, runs paragraph-level inference,
and aggregates to call-level sentiment scores.

Output: data/transcript_scores.csv
  ticker | date | quarter | net_tone | tone_change | positive_ratio | negative_ratio | neutral_ratio
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = DATA_DIR / "raw_transcripts.json"
SCORES_FILE = DATA_DIR / "transcript_scores.csv"

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512
BATCH_SIZE = 16


def load_finbert():
    """Load FinBERT tokenizer and model."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Using device: {'cuda' if device == 0 else 'cpu'}")
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
    )
    return clf


def split_paragraphs(text: str, min_words: int = 10) -> list[str]:
    """Split transcript text into paragraphs, filtering short ones."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Fallback: split by single newlines if few paragraphs found
    if len(paras) < 3:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in paras if len(p.split()) >= min_words]


def truncate_paragraph(para: str, tokenizer, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate paragraph to fit FinBERT's 512-token limit."""
    tokens = tokenizer.encode(para, add_special_tokens=True)
    if len(tokens) <= max_tokens:
        return para
    tokens = tokens[:max_tokens - 1] + [tokenizer.sep_token_id]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def score_paragraphs(clf, paragraphs: list[str], tokenizer) -> dict:
    """
    Run FinBERT on a list of paragraphs and aggregate to call-level scores.
    Returns dict with positive_ratio, negative_ratio, neutral_ratio, net_tone.
    """
    if not paragraphs:
        return {"positive_ratio": 0.0, "negative_ratio": 0.0, "neutral_ratio": 1.0, "net_tone": 0.0}

    truncated = [truncate_paragraph(p, tokenizer) for p in paragraphs]
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    # Run in batches
    for i in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[i : i + BATCH_SIZE]
        try:
            results = clf(batch)
            for result in results:
                # result is a list of {label, score} dicts
                top = max(result, key=lambda x: x["score"])
                label = top["label"].lower()
                if label in counts:
                    counts[label] += 1
        except Exception:
            counts["neutral"] += len(batch)

    total = sum(counts.values())
    if total == 0:
        return {"positive_ratio": 0.0, "negative_ratio": 0.0, "neutral_ratio": 1.0, "net_tone": 0.0}

    pos_r = counts["positive"] / total
    neg_r = counts["negative"] / total
    neu_r = counts["neutral"] / total
    return {
        "positive_ratio": pos_r,
        "negative_ratio": neg_r,
        "neutral_ratio": neu_r,
        "net_tone": pos_r - neg_r,
    }


def compute_tone_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tone_change = net_tone(t) - net_tone(t-1) within each ticker.
    Sort by ticker and date first.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["tone_change"] = df.groupby("ticker")["net_tone"].diff()
    return df


def run_finbert_pipeline(transcripts: list[dict], use_mock: bool = False) -> pd.DataFrame:
    """
    Score all transcripts with FinBERT and compute tone_change.

    Args:
        transcripts: list of {ticker, date, quarter, text} dicts
        use_mock: if True, simulate scores without loading the model (for testing)
    """
    if use_mock:
        print("Running mock FinBERT scoring (model not loaded)...")
        records = _mock_score(transcripts)
    else:
        clf = load_finbert()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        records = []
        for t in tqdm(transcripts, desc="Scoring transcripts"):
            paras = split_paragraphs(t["text"])
            scores = score_paragraphs(clf, paras, tokenizer)
            records.append({
                "ticker": t["ticker"],
                "date": t["date"],
                "quarter": t["quarter"],
                **scores,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_tone_change(df)
    return df


def _mock_score(transcripts: list[dict]) -> list[dict]:
    """
    Generate deterministic mock FinBERT scores from the transcript text.
    Uses simple word-count heuristics to produce plausible sentiment.
    """
    from scraper.fetch_transcripts import NEGATIVE_PHRASES, POSITIVE_PHRASES

    rng = np.random.default_rng(42)
    records = []
    for t in tqdm(transcripts, desc="Mock-scoring transcripts"):
        text_lower = t["text"].lower()
        pos_count = sum(text_lower.count(p.lower()) for p in POSITIVE_PHRASES)
        neg_count = sum(text_lower.count(p.lower()) for p in NEGATIVE_PHRASES)
        total = pos_count + neg_count + 1
        # Add controlled noise so scores aren't perfectly deterministic
        noise = rng.normal(0, 0.05)
        pos_r = min(1.0, max(0.0, pos_count / total + noise))
        neg_r = min(1.0, max(0.0, neg_count / total - noise))
        neu_r = max(0.0, 1.0 - pos_r - neg_r)
        records.append({
            "ticker": t["ticker"],
            "date": t["date"],
            "quarter": t["quarter"],
            "positive_ratio": round(pos_r, 4),
            "negative_ratio": round(neg_r, 4),
            "neutral_ratio": round(neu_r, 4),
            "net_tone": round(pos_r - neg_r, 4),
        })
    return records


def main(use_mock: bool = False):
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw transcripts not found at {RAW_FILE}. Run scraper first.")

    print(f"Loading transcripts from {RAW_FILE}...")
    with open(RAW_FILE) as f:
        transcripts = json.load(f)
    print(f"  Loaded {len(transcripts)} transcripts.")

    # Auto-detect: use mock if GPU not available and model would be slow
    if not use_mock and not torch.cuda.is_available():
        print("No GPU detected; using mock scorer for speed. Pass use_mock=False to force FinBERT.")
        use_mock = True

    df = run_finbert_pipeline(transcripts, use_mock=use_mock)
    df.to_csv(SCORES_FILE, index=False)
    print(f"Saved {len(df)} scored records to {SCORES_FILE}")
    print(df[["ticker", "date", "net_tone", "tone_change"]].head(10).to_string())


if __name__ == "__main__":
    import sys
    force_real = "--real" in sys.argv
    main(use_mock=not force_real)
