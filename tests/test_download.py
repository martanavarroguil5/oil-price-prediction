"""Offline behaviour of the downloader: failures keep the stored raw files."""

import pandas as pd

from oilcast.data import download
from oilcast.data.dataset import align_to_calendar


def test_download_failure_keeps_local_copy(cfg, tmp_paths, monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("sin conexión")

    monkeypatch.setattr(download, "fetch_fred", boom)
    monkeypatch.setattr(download, "fetch_yahoo", boom)
    monkeypatch.setattr(download, "fetch_opec", boom)
    (tmp_paths.raw / "brent.csv").write_text("date,brent\n2026-01-02,70.0\n")
    status = download.download_all(cfg, tmp_paths)
    assert status["brent"].startswith("error") and "se conserva" in status["brent"]
    assert (tmp_paths.raw / "brent.csv").read_text().endswith("70.0\n")


def test_align_never_backfills():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-02", "2020-01-06"]))
    cal = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06", "2020-01-07"])
    out = align_to_calendar(s, pd.DatetimeIndex(cal))
    assert pd.isna(out.iloc[0])
    assert out.tolist()[1:] == [1.0, 2.0, 2.0]


def test_opec_parser():
    xml = b'<BasketList><BasketPrice data="2024-01-02" val="77.5"/></BasketList>'
    s = download.parse_opec_xml(xml)
    assert s.iloc[0] == 77.5
