"""Offline tests (no FinBERT weights / network): tone change, IC, portfolio construction, metrics."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from backtest.longshort_backtest import (
    TC,
    annualized_return,
    construct_portfolio,
    deflated_sharpe_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
)
from signals.ic_analysis import compute_ic_table, spearman_ic


@pytest.fixture(scope="module")
def finbert(monkeypatch_module=None):
    """Import nlp.finbert_scorer with heavy ML dependencies stubbed out."""
    stubs = {}
    for name in ("torch", "transformers"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            stubs[name] = m
            sys.modules[name] = m
    sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
    t = sys.modules["transformers"]
    for attr in ("AutoModelForSequenceClassification", "AutoTokenizer", "pipeline"):
        setattr(t, attr, object)
    try:
        import tqdm  # noqa: F401
    except ImportError:
        tq = types.ModuleType("tqdm")
        tq.tqdm = lambda x, **k: x
        sys.modules["tqdm"] = tq
    import nlp.finbert_scorer as fb

    yield fb
    for name in stubs:
        sys.modules.pop(name, None)


# --------------------------------------------------------------------- NLP
def test_tone_change_is_within_ticker_and_backward_looking(finbert):
    df = pd.DataFrame(
        {
            "ticker": ["B", "A", "A", "B", "A"],
            "date": pd.to_datetime(["2022-02-01", "2022-05-01", "2022-02-01", "2022-05-01", "2022-08-01"]),
            "net_tone": [0.10, 0.30, 0.20, 0.00, 0.25],
        }
    )
    out = finbert.compute_tone_change(df)
    a = out[out["ticker"] == "A"]["tone_change"].tolist()
    b = out[out["ticker"] == "B"]["tone_change"].tolist()
    assert np.isnan(a[0]) and np.isnan(b[0])  # first call per ticker has no change
    assert a[1:] == pytest.approx([0.10, -0.05])
    assert b[1] == pytest.approx(-0.10)


def test_split_paragraphs_filters_short(finbert):
    text = "short one\n\n" + " ".join(["word"] * 12) + "\n\n   \n\n" + " ".join(["more"] * 15)
    paras = finbert.split_paragraphs(text, min_words=10)
    assert len(paras) == 2


# --------------------------------------------------------------------- IC
def test_spearman_ic_perfect_and_small_sample():
    s = pd.Series(np.arange(20.0))
    ic, t = spearman_ic(s, s * 3)
    assert ic == pytest.approx(1.0) and t > 100
    assert np.isnan(spearman_ic(s.iloc[:5], s.iloc[:5])[0])


def test_ic_table_only_uses_available_horizons():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"tone_change": rng.normal(size=60)})
    df["fwd_5d"] = df["tone_change"] + rng.normal(scale=0.5, size=60)
    out = compute_ic_table(df)
    assert out["horizon_days"].tolist() == [5]
    assert out["ic"].iloc[0] > 0.5


# --------------------------------------------------------------- portfolio
def _month(date, n=30, seed=0, edge=0.05):
    rng = np.random.default_rng(seed)
    tone = np.linspace(-1, 1, n)
    return pd.DataFrame({"date": pd.Timestamp(date), "tone_change": tone, "fwd_20d": edge * tone + rng.normal(0, 0.001, n)})


def test_portfolio_long_top_short_bottom_and_costs():
    scores = pd.concat([_month("2022-01-31", seed=1), _month("2022-02-28", seed=2)])
    port = construct_portfolio(scores)
    assert len(port) == 2
    assert (port["long_ret"] > port["short_ret"]).all()
    np.testing.assert_allclose(port["net_ret"], port["gross_ret"] - 4 * TC)
    assert (port["n_long"] >= 3).all() and (port["n_short"] >= 3).all()


def test_portfolio_skips_thin_months_and_nans():
    thin = _month("2022-03-31", n=8)
    full = _month("2022-04-29", n=30)
    full.loc[:5, "tone_change"] = np.nan
    port = construct_portfolio(pd.concat([thin, full]))
    assert port["date"].tolist() == [pd.Timestamp("2022-04-29")]


def test_portfolio_ignores_returns_when_ranking():
    """Ranking depends only on tone_change; changing realised returns cannot change who is held."""
    base = _month("2022-01-31", seed=3)
    alt = base.copy()
    alt["fwd_20d"] = alt["fwd_20d"].values[::-1]
    a, b = construct_portfolio(base), construct_portfolio(alt)
    assert (a["n_long"].iloc[0], a["n_short"].iloc[0]) == (b["n_long"].iloc[0], b["n_short"].iloc[0])


# ----------------------------------------------------------------- metrics
def test_basic_metrics():
    r = pd.Series([0.10, -0.05, 0.02, 0.03])
    assert hit_rate(r) == 0.75
    assert annualized_return(pd.Series([0.01] * 12)) == pytest.approx(1.01**12 - 1)
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std() * np.sqrt(12))
    equity = (1 + pd.Series([0.5, 0.33333333, -0.5])).cumprod()
    assert max_drawdown(pd.Series([0.5, -0.5])) == pytest.approx(-0.5)


def test_deflated_sharpe_is_finite_for_high_sharpe_and_monotone_in_trials():
    """Regression: annualised SR was fed into the variance-inflation term, giving NaN/garbage."""
    d1 = deflated_sharpe_ratio(3.0, n_trials=1, n_obs=72, skew=-0.5, kurt=5.0)
    d10 = deflated_sharpe_ratio(3.0, n_trials=10, n_obs=72, skew=-0.5, kurt=5.0)
    d100 = deflated_sharpe_ratio(3.0, n_trials=100, n_obs=72, skew=-0.5, kurt=5.0)
    assert 0 <= d100 <= d10 <= d1 <= 1
    assert np.isfinite([d1, d10, d100]).all()


def test_deflated_sharpe_zero_sharpe_is_half_with_one_trial_and_longer_history_helps():
    assert deflated_sharpe_ratio(0.0, 1, 100) == pytest.approx(0.5)
    short = deflated_sharpe_ratio(1.0, 10, 36)
    long_ = deflated_sharpe_ratio(1.0, 10, 144)
    assert long_ > short
    assert np.isnan(deflated_sharpe_ratio(1.0, 10, 1))
