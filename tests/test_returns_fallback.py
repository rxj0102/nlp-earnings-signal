import pandas as pd
import pytest

from signals import ic_analysis as ic


def _df():
    return pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "date": pd.to_datetime(["2021-01-05", "2021-01-06"]),
         "tone_change": [0.1, -0.1]}
    )


def test_empty_download_raises_by_default(monkeypatch):
    monkeypatch.setattr(ic.yf, "download", lambda *a, **k: pd.DataFrame({"Close": []}))
    with pytest.raises(RuntimeError):
        ic.fetch_forward_returns(_df())


def test_synthetic_is_opt_in_and_a_dataframe(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(ic.yf, "download", boom)
    with pytest.raises(RuntimeError):
        ic.fetch_forward_returns(_df())
    out = ic.fetch_forward_returns(_df(), allow_synthetic=True)
    assert isinstance(out, pd.DataFrame) and "fwd_5d" in out.columns and len(out) == 2
