"""Exercise the same HTTP interface and saved forecasts used by the HTML client."""

import threading
from http.server import ThreadingHTTPServer

import pytest
import requests

from oilcast.config import load_config
from oilcast.paths import get_paths
from oilcast.web import Observatory, handler


@pytest.fixture
def server():
    cfg = load_config()
    service = Observatory(cfg, get_paths(cfg))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler(service)) as httpd:
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_port}", service
        finally:
            httpd.shutdown()
            thread.join()


def test_frontend_and_forecasts(server):
    url, service = server
    html = requests.get(url, timeout=5)
    assert html.ok and 'id="lab-stage"' in html.text
    assert requests.get(url + "/vendor/three.js", timeout=5).ok
    if not (service.paths.models / "metadata.json").exists():
        pytest.skip("production artifacts absent")
    state = requests.get(url + "/api/state", timeout=30).json()
    assert set(state["forecasts"]) == {"5", "63", "252"}
    for horizon, fc in state["forecasts"].items():
        lo, hi = fc["intervals"]["80"]
        assert lo <= fc["price"] <= hi and fc["date"] > state["date"]
        assert fc["explanation"]["reliability"]
        if state["comparison"]:
            comparison = [r for r in state["comparison"] if str(r["horizon"]) == horizon]
            assert all(isinstance(r["r2"], float) for r in comparison)
            rows = requests.get(url + "/api/validation/rnn/" + horizon, timeout=5).json()
            assert len(rows) == comparison[0]["n_test"] and rows[0][0] < rows[-1][0]


def test_data_controls_are_separate(server, monkeypatch):
    url, service = server
    actions = []
    monkeypatch.setattr(service, "start", lambda *args: actions.append(args) or True)
    assert requests.post(url + "/api/refresh", json={}, timeout=3).status_code == 202
    assert requests.post(url + "/api/retrain", json={"mode": "full"}, timeout=3).status_code == 202
    assert actions == [("refresh", "fast"), ("retrain", "full")]
    response = requests.post(
        url + "/api/retrain", json={}, headers={"Origin": "https://example.org"}, timeout=3
    )
    assert response.status_code == 403


def test_historical_replay_http_requires_reveal_and_exposes_return_metrics(server):
    url, service = server
    if not (service.paths.results / "predictions/rnn_h63_ensemble.csv").exists():
        pytest.skip("saved predictions absent")
    available = requests.get(url + "/api/replay/rnn/63", timeout=5).json()["origins"]
    route = url + "/api/replay/rnn/63/" + available[0]
    before = requests.get(route, timeout=5).json()
    assert before["revealed"] is False and "outcome" not in before
    after = requests.get(route + "/reveal", timeout=5).json()
    assert after["revealed"] and "actual" in after["outcome"]
    assert requests.get(url + "/api/replay/rnn/63/2022-02-30", timeout=5).status_code == 404
    assert requests.get(url + "/api/replay/unknown/63", timeout=5).status_code == 404
    state = requests.get(url + "/api/state", timeout=30).json()
    assert {row["horizon"] for row in state["returnMetrics"]} == {5, 63, 252}
    if state.get("intervalStudy") and not state["intervalStudy"]["stale"]:
        for h, preview in state["intervalStudy"]["latest"].items():
            assert preview["published"] == state["forecasts"][h]["intervals"]
            lo, hi = preview["experimental"]["80"]
            assert 0 < lo <= state["forecasts"][h]["price"] <= hi


def test_saved_diagnostics_are_exposed_as_images(server):
    url, service = server
    if not (service.paths.figures / "diagnostics.json").exists():
        pytest.skip("diagnostic plots absent")
    state = requests.get(url + "/api/state", timeout=30).json()
    assert state["diagnostics"]["candles"]["sessions"] > 0
    for name in (
        "brent_velas.png",
        "violines_precio_h63.png",
        "violines_retorno_h252.png",
        "dispersion_rnn.png",
    ):
        response = requests.get(url + "/figures/" + name, timeout=5)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_refresh_publishes_new_day_with_saved_weights(tmp_path, monkeypatch):
    import shutil

    from oilcast.data.dataset import load_dataset
    from oilcast.web import snapshot

    cfg = load_config()
    source, copy = get_paths(cfg), get_paths(cfg, root=tmp_path)
    if not (source.models / "metadata.json").exists():
        pytest.skip("production artifacts absent")
    copy.ensure()
    shutil.copytree(source.raw, copy.raw, dirs_exist_ok=True)
    shutil.copytree(source.models, copy.models, dirs_exist_ok=True)
    previous = load_dataset(source).iloc[:-1]
    previous.to_csv(copy.dataset_csv)
    models_before = {p.name: p.read_bytes() for p in copy.models.iterdir() if p.is_file()}
    monkeypatch.setattr("oilcast.web.download_all", lambda cfg, paths: {"brent": "ok (local)"})
    service = Observatory(cfg, copy)
    service._run("refresh", "fast")
    assert service.job["status"] == "done"
    assert load_dataset(copy).index[-1] > previous.index[-1]
    assert models_before == {p.name: p.read_bytes() for p in copy.models.iterdir() if p.is_file()}
    result = snapshot(cfg, copy)
    assert all(fc["date"] > result["date"] for fc in result["forecasts"].values())

    published = copy.dataset_csv.read_bytes()
    monkeypatch.setattr("oilcast.web.download_all", lambda cfg, paths: {"brent": "error"})
    service._run("refresh", "fast")
    assert service.job["status"] == "error" and copy.dataset_csv.read_bytes() == published
