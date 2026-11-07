"""
Long-short backtest on FinBERT tone_change signal.

Strategy: monthly rebalancing, long top-3 deciles, short bottom-3 deciles.
Transaction costs: 10bps one-way.
Reports: annualized return, Sharpe, max drawdown, hit rate, Deflated Sharpe Ratio.
Out-of-sample split: train 2018-2020, test 2021-2023.

Output: outputs/strategy_performance.csv
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SCORES_FILE = DATA_DIR / "transcript_scores.csv"
RETURNS_FILE = DATA_DIR / "forward_returns.csv"
PERF_FILE = OUTPUTS_DIR / "strategy_performance.csv"

TC_BPS = 10  # one-way transaction cost in basis points
TC = TC_BPS / 10_000

TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def construct_portfolio(scores_monthly: pd.DataFrame, returns_col: str = "fwd_20d") -> pd.DataFrame:
    """
    For each rebalance date, rank tone_change into deciles.
    Long: top 3 deciles. Short: bottom 3 deciles.
    Equal-weight within each leg.
    Returns a time series of monthly portfolio returns.
    """
    port_returns = []

    for date, group in scores_monthly.groupby("date"):
        group = group.dropna(subset=["tone_change", returns_col])
        if len(group) < 10:
            continue

        group = group.copy()
        group["decile"] = pd.qcut(group["tone_change"], q=10, labels=False, duplicates="drop") + 1

        long_leg = group[group["decile"] >= 8]
        short_leg = group[group["decile"] <= 3]

        if len(long_leg) == 0 or len(short_leg) == 0:
            continue

        long_ret = long_leg[returns_col].mean()
        short_ret = short_leg[returns_col].mean()

        # Net return: long - short - 2x TC (entry + exit approximated as 2x one-way)
        gross = long_ret - short_ret
        net = gross - 2 * TC * 2  # round-trip for both legs

        port_returns.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "gross_ret": gross,
            "net_ret": net,
            "n_long": len(long_leg),
            "n_short": len(short_leg),
        })

    return pd.DataFrame(port_returns).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def annualized_return(rets: pd.Series, periods_per_year: float = 12) -> float:
    """Compound annualized return from period returns."""
    cum = (1 + rets).prod()
    n = len(rets)
    return cum ** (periods_per_year / n) - 1


def sharpe_ratio(rets: pd.Series, rf: float = 0.0, periods_per_year: float = 12) -> float:
    excess = rets - rf / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def max_drawdown(rets: pd.Series) -> float:
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return dd.min()


def hit_rate(rets: pd.Series) -> float:
    return (rets > 0).mean()


def deflated_sharpe_ratio(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    periods_per_year: float = 12,
) -> float:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    ``sr`` is the ANNUALISED Sharpe; ``kurt`` is raw (non-excess) kurtosis.
    The estimator variance (skew/kurtosis inflation) must use the per-period
    Sharpe, so ``sr`` is de-annualised internally.  Returns P(true SR > 0)
    after deflating for the best of ``n_trials`` independent trials.
    """
    if n_obs <= 1:
        return np.nan
    n_trials = max(int(n_trials), 1)
    sr_p = sr / np.sqrt(periods_per_year)

    # Expected maximum of n_trials standard-normal Sharpe estimates
    if n_trials == 1:
        expected_max = 0.0
    else:
        gamma = np.euler_gamma
        expected_max = (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials) + gamma * stats.norm.ppf(
            1 - 1 / (n_trials * np.e)
        )

    # Variance inflation for non-normal returns (per-period SR)
    infl = max(1 - skew * sr_p + (kurt - 1) / 4 * sr_p**2, 1e-12)

    # z = (SR_hat - SR0) / sd(SR_hat), with SR0 = expected_max * sd(SR_hat)
    z = sr_p * np.sqrt(n_obs - 1) / np.sqrt(infl) - expected_max
    return float(stats.norm.cdf(z))


def compute_metrics(port: pd.DataFrame, label: str, n_trials: int = 10) -> dict:
    rets = port["net_ret"]
    sr = sharpe_ratio(rets)
    n = len(rets)
    skew = float(stats.skew(rets))
    kurt = float(stats.kurtosis(rets, fisher=False))
    dsr = deflated_sharpe_ratio(sr, n_trials=n_trials, n_obs=n, skew=skew, kurt=kurt)

    return {
        "period": label,
        "n_months": n,
        "ann_return": round(annualized_return(rets), 4),
        "sharpe": round(sr, 3),
        "max_drawdown": round(max_drawdown(rets), 4),
        "hit_rate": round(hit_rate(rets), 3),
        "deflated_sharpe": round(dsr, 3),
        "mean_monthly_ret": round(rets.mean(), 4),
        "vol_monthly": round(rets.std(), 4),
    }


# ---------------------------------------------------------------------------
# Momentum integration / incremental IC
# ---------------------------------------------------------------------------

def compute_momentum_signal(scores: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """
    Approximate price momentum using net_tone as a proxy (no prices needed).
    Incremental IC shows how tone_change adds to momentum.
    """
    scores = scores.sort_values(["ticker", "date"]).copy()
    scores["momentum_proxy"] = scores.groupby("ticker")["net_tone"].transform(
        lambda x: x.rolling(lookback_months, min_periods=3).mean()
    )
    return scores


def incremental_ic_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IC of tone_change controlling for momentum (residual IC).
    Returns table comparing standalone vs. incremental IC.
    """
    from scipy.stats import spearmanr

    results = []
    for h in [5, 10, 20]:
        ret_col = f"fwd_{h}d"
        if ret_col not in merged.columns:
            continue
        data = merged[["tone_change", "momentum_proxy", ret_col]].dropna()
        if len(data) < 20:
            continue

        ic_alone, _ = spearmanr(data["tone_change"], data[ret_col])
        ic_mom, _ = spearmanr(data["momentum_proxy"], data[ret_col])

        # Residualize tone_change on momentum (rank regression)
        from scipy.stats import rankdata
        r_tone = rankdata(data["tone_change"])
        r_mom = rankdata(data["momentum_proxy"])
        r_ret = rankdata(data[ret_col])
        # OLS residuals of tone on momentum
        slope = np.cov(r_tone, r_mom)[0, 1] / np.var(r_mom)
        resid = r_tone - slope * r_mom
        ic_incremental, _ = spearmanr(resid, r_ret)

        results.append({
            "horizon": h,
            "ic_tone_standalone": round(ic_alone, 4),
            "ic_momentum": round(ic_mom, 4),
            "ic_tone_incremental": round(ic_incremental, 4),
            "incremental_alpha": round(ic_incremental / max(1e-6, abs(ic_alone)), 3),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Equity curve plot
# ---------------------------------------------------------------------------

def plot_equity_curve(train_port: pd.DataFrame, test_port: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for ax, port, label, color in [
        (axes[0], train_port, "Train (2018-2020)", "#2166ac"),
        (axes[1], test_port, "Test (2021-2023)", "#d6604d"),
    ]:
        cum = (1 + port["net_ret"]).cumprod()
        ax.plot(port["date"], cum, color=color, linewidth=2, label="L/S Strategy")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(port["date"], 1, cum, alpha=0.15, color=color)
        ax.set_title(f"Long-Short Strategy Equity Curve — {label}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return", fontsize=11)
        ax.legend()
        sns.despine(ax=ax)

    plt.tight_layout()
    path = OUTPUTS_DIR / "equity_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved equity curve to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SCORES_FILE.exists() or not RETURNS_FILE.exists():
        print("Run scraper, nlp/finbert_scorer.py, and signals/ic_analysis.py first.")
        sys.exit(1)

    scores = pd.read_csv(SCORES_FILE, parse_dates=["date"])
    returns = pd.read_csv(RETURNS_FILE, parse_dates=["date"])
    merged = scores.merge(returns, on=["ticker", "date"], how="inner")

    # Resample to monthly: use latest transcript per ticker per month
    merged["year_month"] = merged["date"].dt.to_period("M")
    monthly = (
        merged.sort_values("date")
        .groupby(["ticker", "year_month"])
        .last()
        .reset_index()
    )
    monthly["date"] = monthly["year_month"].dt.to_timestamp()

    # Add momentum proxy
    monthly = compute_momentum_signal(monthly)

    # Split train/test
    train = monthly[monthly["date"] <= TRAIN_END].copy()
    test = monthly[monthly["date"] >= TEST_START].copy()

    print(f"Train: {len(train)} obs | Test: {len(test)} obs")

    train_port = construct_portfolio(train)
    test_port = construct_portfolio(test)

    if len(train_port) == 0 or len(test_port) == 0:
        print("Insufficient data for backtest. Check data coverage.")
        return

    train_metrics = compute_metrics(train_port, "train_2018_2020")
    test_metrics = compute_metrics(test_port, "test_2021_2023")

    perf_df = pd.DataFrame([train_metrics, test_metrics])
    perf_df.to_csv(PERF_FILE, index=False)
    print(f"\nSaved performance to {PERF_FILE}")
    print("\n--- Strategy Performance ---")
    print(perf_df.to_string(index=False))

    # Incremental IC
    inc_ic = incremental_ic_analysis(monthly)
    if not inc_ic.empty:
        inc_path = OUTPUTS_DIR / "incremental_ic.csv"
        inc_ic.to_csv(inc_path, index=False)
        print(f"\n--- Incremental IC (tone_change vs. momentum) ---")
        print(inc_ic.to_string(index=False))

    # Equity curve
    plot_equity_curve(train_port, test_port)


if __name__ == "__main__":
    main()
