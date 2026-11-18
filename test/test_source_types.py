"""离线检查数据源整理逻辑，不请求 SEC、Wikipedia 或 Yahoo。"""

import json
from datetime import date
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
import yfinance as yf

from event_finance_frame.sources import universe
from event_finance_frame.sources.fundamental import read_company_facts
from event_finance_frame.sources.prices import get_prices


@pytest.mark.parametrize("has_facts", [False, True])
def test_company_facts_numeric_types(tmp_path: Path, has_facts: bool):
    fact = {
        "end": "2023-12-31",
        "val": 123.5,
        "accn": "test-filing",
        "form": "10-K",
        "filed": "2024-02-01",
    }
    payload = {"facts": {}}
    if has_facts:
        payload["facts"] = {
            "us-gaap": {
                "Assets": {"units": {"USD": [fact]}},
                "Revenues": {"units": {"USD": [{**fact, "start": "2023-01-01"}]}},
            }
        }
    path = tmp_path / "facts.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("CIK0000000001.json", json.dumps(payload))

    period, instant = read_company_facts(path, ["0000000001"])

    for frame in (period, instant):
        assert len(frame) == int(has_facts)
        assert str(frame["val"].dtype) == "float64"
        assert str(frame["fy"].dtype) == "Int64"
        if has_facts:
            assert frame["val"].tolist() == [123.5]
            assert bool(frame["fy"].isna().all())
    assert "start_date" in period.columns
    assert "start_date" not in instant.columns


def test_universe_normalizes_and_sorts_tickers(monkeypatch: pytest.MonkeyPatch):
    html = b"""<table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sub-Industry</th><th>CIK</th></tr>
      <tr><td>BRK.B</td><td>Berkshire</td><td>Insurance</td><td>1067983</td></tr>
      <tr><td>AAPL</td><td>Apple</td><td>Hardware</td><td>0000320193</td></tr>
    </table>"""
    monkeypatch.setattr(universe, "urlopen", lambda *args, **kwargs: BytesIO(html))

    result = universe.fetch_sp500_universe(snapshot_date=date(2026, 11, 17))

    assert result.columns.tolist() == universe.UNIVERSE_COLUMNS
    assert result["ticker"].tolist() == ["AAPL", "BRK-B"]
    assert result["cik"].tolist() == ["0000320193", "0001067983"]
    assert result["name"].tolist() == ["Apple", "Berkshire"]
    assert (result["snapshot_date"] == pd.Timestamp("2026-11-17")).all()
    assert bool(result["exchange"].isna().all())
    assert result.index.tolist() == [0, 1]


def test_prices_uses_public_download_and_keeps_tickers(monkeypatch: pytest.MonkeyPatch):
    raw = pd.DataFrame(
        [[10.0, 9.0, 100, 20.0, 19.0, 200]],
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
        columns=pd.MultiIndex.from_tuples(
            [
                ("AAA", "Close"),
                ("AAA", "Adj Close"),
                ("AAA", "Volume"),
                ("BBB", "Close"),
                ("BBB", "Adj Close"),
                ("BBB", "Volume"),
            ]
        ),
    )
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return raw.copy()

    monkeypatch.setattr(yf, "download", fake_download)

    result = get_prices(["AAA", "BBB"], "2024-01-01", "2024-01-03", progress=False)

    assert len(calls) == 1
    assert "session" not in calls[0][1]
    assert result.index.names == ["date", "ticker"]
    assert result["close"].tolist() == [10.0, 20.0]
    assert result["adj_close"].tolist() == [9.0, 19.0]
    assert result["volume"].tolist() == [100, 200]
    assert bool(result["open"].isna().all())
