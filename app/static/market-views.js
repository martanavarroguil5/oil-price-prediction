const $ = id => document.getElementById(id);
const number = (value, n = 2) => Number(value).toLocaleString("es-ES", {
  minimumFractionDigits: n, maximumFractionDigits: n,
});
const date = s => new Intl.DateTimeFormat("es-ES", {
  day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
}).format(new Date(s));
const labels = { naive: "Ingenuo", rnn: "RNN simple", lstm: "LSTM", gru: "GRU" };
const node = (tag, text) => { const element = document.createElement(tag); element.textContent = text; return element; };

export function renderMarketViews(state, h) {
  renderIntervalStudy(state, h);
  const meta = state.diagnostics;
  $("candles-panel").hidden = !meta;
  $("violin-panel").hidden = !meta;
  if (meta) {
    const unit = $("violin-unit").value, suffix = "?v=" + meta.fingerprint;
    const candleSource = "/figures/brent_velas.png" + suffix;
    if ($("candles-image").getAttribute("src") !== candleSource) $("candles-image").src = candleSource;
    $("candles-download").href = candleSource;
    $("candles-period").textContent = meta.candles.source + " · " + meta.candles.sessions +
      " sesiones · " + date(meta.candles.first) + "–" + date(meta.candles.last);
    const violinSource = "/figures/violines_" + unit + "_h" + h + ".png" + suffix;
    if ($("violin-image").getAttribute("src") !== violinSource) $("violin-image").src = violinSource;
    $("violin-download").href = violinSource;
    $("violin-image").alt = "Distribución de errores de ingenuo, RNN, LSTM y GRU a " + h +
      " sesiones, en " + (unit === "precio" ? "USD por barril" : "puntos porcentuales de rentabilidad");
    const sample = meta.samples[String(h)];
    $("violin-period").textContent = sample.n + " orígenes por modelo · " + date(sample.first) +
      "–" + date(sample.last) + ". Previsiones diarias solapadas; las observaciones no son independientes.";
  }
  const rows = state.comparison.filter(row => row.horizon === h);
  $("interval-body").replaceChildren(...rows.map(metrics => {
    const row = document.createElement("tr");
    if (metrics.model === state.model.arch) row.className = "chosen";
    for (const text of [labels[metrics.model], number(metrics.coverage80 * 100, 1) + " %",
      number(metrics.width80), number(metrics.winkler80)]) row.append(node("td", text));
    return row;
  }));
}

function renderIntervalStudy(state, h) {
  const study = state.intervalStudy;
  $("interval-study-panel").hidden = !study;
  if (!study) return;
  const rows = study.comparison.filter(row => row.horizon === h);
  const method = name => name === "current" ? "Publicado" :
    name.startsWith("residual") ? "Errores de la mediana" :
    name.startsWith("scaled") ? "Corrección según anchura" : "CQR con ventana reciente";
  $("interval-study-body").replaceChildren(...rows.map(metrics => {
    const row = document.createElement("tr");
    if (metrics.arch === state.model.arch) row.className = "chosen";
    for (const value of [labels[metrics.arch], method(metrics.candidate),
      number(metrics.current_coverage80*100,1)+" → "+number(metrics.coverage80*100,1)+" %",
      number(metrics.current_width80)+" → "+number(metrics.width80),
      number(metrics.current_winkler80)+" → "+number(metrics.winkler80),
      (metrics.skill80 > 0 ? "+" : "")+number(metrics.skill80*100)+" %"])
      row.append(node("td",value));
    return row;
  }));
  const latest = study.latest[String(h)];
  $("interval-study-latest").textContent = study.stale ?
    "Las predicciones guardadas han cambiado. La comparación requiere una nueva evaluación antes de mostrar intervalos actuales." :
    latest ? "Previsión actual de "+labels[study.arch]+" · "+date(latest.origin)+" → "+date(latest.target)+
      ". Rango publicado del 80 %: "+latest.published["80"].map(v=>number(v)).join("–")+
      " USD/barril. Experimental: "+latest.experimental["80"].map(v=>number(v)).join("–")+" USD/barril." :
      "No hay una previsión actual disponible para esta comparación.";
}
