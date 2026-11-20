"""La web estática (GitHub Pages) lleva los mismos datos que la API del servidor local."""

import json

import pytest

from oilcast.config import load_config
from oilcast.paths import get_paths
from oilcast.replay import replay_case
from oilcast.static_site import STATIC_META, export_static


def test_export_matches_the_server(tmp_path):
    cfg = load_config()
    paths = get_paths(cfg)
    if not (paths.models / "metadata.json").exists():
        pytest.skip("production artifacts absent")
    out = export_static(cfg, paths, tmp_path / "site")

    html = (out / "index.html").read_text(encoding="utf-8")
    assert STATIC_META in html and 'src="/' not in html and 'href="/' not in html
    assert (out / "api.js").exists() and (out / "vendor" / "three.js").exists()
    state = json.loads((out / "api" / "state.json").read_text(encoding="utf-8"))
    assert set(state["forecasts"]) == {"5", "63", "252"}

    cases = json.loads((out / "api" / "replay" / "rnn_63.json").read_text(encoding="utf-8"))
    origin, target, _, price, median, lo, hi, actual = cases["rows"][100]
    case = replay_case(paths, "rnn", 63, origin, reveal=True)
    assert target == case["targetDate"]
    assert price == pytest.approx(case["price"], abs=1e-4)
    assert median == pytest.approx(case["median"], abs=1e-4)
    assert [lo, hi] == pytest.approx(case["interval"], abs=1e-4)
    assert actual == pytest.approx(case["outcome"]["actual"], abs=1e-4)
    assert len(json.loads((out / "api" / "validation" / "naive_252.json").read_text())) > 0


def test_export_refuses_to_overwrite_the_project(tmp_path):
    cfg = load_config()
    paths = get_paths(cfg)
    with pytest.raises(ValueError):
        export_static(cfg, paths, paths.root)
