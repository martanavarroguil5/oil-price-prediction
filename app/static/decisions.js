import { calculatePlan } from "./planning.js";
import { replayChart } from "./charts.js";

const $ = id => document.getElementById(id);
const node = (tag, text, cls) => {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (cls) element.className = cls;
  return element;
};
const num = (v, digits = 0) => Number(v).toLocaleString("es-ES", {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
});
const signed = (v, digits = 1) => (v > 0 ? "+" : "") + num(v, digits);
const day = s => new Intl.DateTimeFormat("es-ES", {
  day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
}).format(new Date(s));
const names = { rnn: "RNN simple", lstm: "LSTM", gru: "GRU" };
const horizons = { 5: "1 semana", 63: "3 meses", 252: "1 año" };
let access, referenceEdited = false, lastPlan, replaySerial = 0, replayRecord, replayKey;
const datesCache = new Map();

async function get(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let message;
    try { message = (await response.json()).error; } catch {}
    throw Error(message || "No se ha podido cargar la previsión histórica.");
  }
  return response.json();
}

export function bindDecisions(options) {
  access = options;
  $("plan-form").onsubmit = event => event.preventDefault();
  for (const id of ["plan-volume", "plan-budget", "plan-reference", "plan-fee", "plan-share", "plan-shock"])
    $(id).addEventListener("input", () => {
      if (id === "plan-reference") referenceEdited = true;
      if (access.state()) renderPlanning(access.state(), access.horizon());
    });
  $("export-plan").onclick = exportPlan;
  $("replay-model").onchange = () => renderReplay();
  $("replay-date").onchange = () => renderReplay();
  $("replay-previous").onclick = () => stepReplay(-1);
  $("replay-next").onclick = () => stepReplay(1);
  $("replay-reveal").onclick = revealReplay;
}

export function renderDecisions(state, h) {
  if (!referenceEdited) $("plan-reference").value = state.history.at(-1)[1].toFixed(2);
  renderEvidence(state, h);
  renderPlanning(state, h);
}

function renderEvidence(state, h) {
  const app = state.comparison.filter(row => row.model === state.model.arch);
  $("horizon-evidence").replaceChildren(...[5, 63, 252].map(days => {
    const metrics = app.find(row => row.horizon === days);
    const card = node("button", null, "horizon-evidence");
    card.type = "button";
    card.setAttribute("aria-pressed", String(days === h));
    card.append(node("span", horizons[days], "eyebrow"));
    if (metrics) {
      card.append(node("strong", signed(metrics.skill_mae * 100) + " %",
        metrics.skill_mae < 0 ? "amber" : ""));
      card.append(node("span", metrics.skill_mae < 0 ? "Sin ventaja histórica" :
        metrics.skill_mae < 0.02 ? "Ganancia histórica marginal" : "Ganancia histórica modesta"));
    } else card.append(node("strong", "Sin evaluación"));
    card.append(node("small", "Reducción del MAE frente al precio constante"));
    card.onclick = () => access.selectHorizon(days);
    return card;
  }));
  const metrics = app.find(row => row.horizon === h);
  const naive = state.comparison.find(row => row.horizon === h && row.model === "naive");
  $("evidence-title").textContent = "A " + horizons[h];
  $("mini-spec").textContent = state.model.arch_label + " · " + state.model.window + " sesiones";
  const detail = $("evidence-detail");
  detail.replaceChildren();
  if (!metrics || !naive) { detail.textContent = "No hay evaluación histórica disponible."; return; }
  for (const [label, value] of [
    ["Error medio de la red", num(metrics.mae, 2) + " USD/bbl"],
    ["Error del precio constante", num(naive.mae, 2) + " USD/bbl"],
    ["Cobertura observada del intervalo 80 %", num(metrics.coverage80 * 100, 1) + " %"],
  ]) {
    const item = node("div"); item.append(node("span", label), node("strong", value)); detail.append(item);
  }
  detail.append(node("p", "Periodo " + metrics.first_origin.slice(0, 4) + "–" +
    metrics.last_origin.slice(0, 4) + ". Evaluación retrospectiva; la cobertura pasada no garantiza la siguiente previsión.", "footnote"));
}

function renderPlanning(state, h) {
  const fields = ["plan-volume", "plan-budget", "plan-reference", "plan-fee"];
  const valid = fields.every(id => $(id).value.trim() !== "" && $(id).checkValidity());
  $("plan-error").hidden = valid;
  $("plan-error").textContent = valid ? "" : "Completa los campos con valores válidos: volumen y precio positivos; presupuesto y coste adicional no negativos.";
  $("export-plan").disabled = !valid;
  $("budget-track").parentElement.hidden = !valid;
  if (!valid) {
    lastPlan = null;
    for (const id of ["plan-central", "plan-range", "plan-margin"]) $(id).textContent = "—";
    $("strategy-body").replaceChildren();
    $("strategy-reading").textContent = "Completa los datos para comparar los escenarios.";
    return;
  }
  const fc = state.forecasts[h], share = Number($("plan-share").value) / 100;
  const inputs = {
    volume: Number($("plan-volume").value), budget: Number($("plan-budget").value),
    reference: Number($("plan-reference").value), fee: Number($("plan-fee").value),
    median: fc.price, low: fc.intervals["80"][0], high: fc.intervals["80"][1],
    share, shock: Number($("plan-shock").value),
  };
  const result = calculatePlan(inputs), split = result.strategies.find(s => s.id === "split");
  lastPlan = { inputs, result, origin: state.date, target: fc.date, horizon: h };
  $("share-label").textContent = num(share * 100) + " %";
  $("shock-label").textContent = signed(inputs.shock, 0) + " %";
  $("stress-heading").textContent = "Manual " + signed(inputs.shock, 0) + " %";
  $("plan-central").textContent = num(split.costs.central);
  $("plan-range").textContent = num(split.costs.low) + " – " + num(split.costs.high);
  $("plan-margin-label").textContent = split.margin < 0 ? "Exceso sobre presupuesto" : "Margen bajo presupuesto";
  $("plan-margin").textContent = num(Math.abs(split.margin));
  $("plan-margin").className = split.margin < 0 ? "amber" : "mint";
  const min = Math.min(split.costs.low, inputs.budget) * 0.95;
  const max = Math.max(split.costs.high, inputs.budget, min + 1) * 1.05;
  const position = v => (v - min) / (max - min) * 100;
  $("budget-band").style.left = position(split.costs.low) + "%";
  $("budget-band").style.width = (position(split.costs.high) - position(split.costs.low)) + "%";
  $("budget-center").style.left = position(split.costs.central) + "%";
  $("budget-limit").style.left = position(inputs.budget) + "%";
  $("plan-reading").textContent = "Con este reparto, cada subida de 1 USD/barril en el precio final aumenta el coste en " +
    num(split.sensitivity) + " USD. El escenario central corresponde al " + day(fc.date) + ".";
  $("plan-assumptions").textContent = "Coste = volumen × [parte al precio de referencia + parte al precio final + coste adicional unitario]. " +
    "Los escenarios finales proceden de la previsión a " + h + " sesiones, emitida con el último dato del " + day(state.date) + ".";
  $("strategy-body").replaceChildren(...result.strategies.map(strategy => {
    const row = node("tr", null, strategy.id === "split" ? "chosen" : "");
    const label = node("td");
    label.append(node("strong", strategy.label), node("small", num(strategy.locked * 100) + "% ahora · " +
      num((1 - strategy.locked) * 100) + "% al final"));
    row.append(label);
    for (const key of ["low", "central", "high", "stress"]) {
      const cost = strategy.costs[key], cell = node("td", num(cost));
      if (cost > inputs.budget) { cell.className = "over-budget"; cell.append(node("small", "supera presupuesto")); }
      row.append(cell);
    }
    return row;
  }));
  $("strategy-reading").textContent = "Precio final manual: " + num(result.prices.stress, 2) +
    " USD/barril. Las cantidades de la tabla son costes totales en USD; las marcas ámbar indican que exceden tu presupuesto.";
}

function exportPlan() {
  if (!lastPlan) return;
  const { inputs, result, origin, target, horizon } = lastPlan;
  const lines = ["ESTRATO · ESCENARIO DE PRESUPUESTO", "", "Datos de origen: " + origin,
    "Fecha objetivo aproximada: " + target + " (" + horizon + " sesiones)",
    "Volumen: " + inputs.volume + " barriles equivalentes", "Presupuesto: " + inputs.budget + " USD",
    "Referencia: " + inputs.reference + " USD/barril", "Coste adicional: " + inputs.fee + " USD/barril",
    "Reparto: " + inputs.share * 100 + "% al precio de referencia",
    "Precios finales inferior / central / superior: " + [inputs.low, inputs.median, inputs.high].map(v => num(v, 2)).join(" / "),
    "Escenario manual: " + inputs.shock + "% sobre la mediana", ""];
  for (const strategy of result.strategies) lines.push(strategy.label + ": " +
    Object.entries(strategy.costs).map(([k, v]) => k + "=" + num(v, 2) + " USD").join("; "));
  lines.push("", "Simulación por escenarios, sin probabilidades asignadas al escenario manual ni estrategia recomendada.",
    "La referencia spot no es una oferta ejecutable. El rango nominal del 80% no garantiza cobertura futura.",
    "No incluye financiación, almacenamiento ni derivados. Solo se incluye el coste adicional indicado.");
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" }));
  const link = node("a"); link.href = url; link.download = "estrato-escenario-" + horizon + "-sesiones.txt";
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function renderReturns(state, h) {
  const rows = (state.returnMetrics || []).filter(row => row.horizon === h);
  $("return-body").replaceChildren(...rows.map(metrics => {
    const row = node("tr", null, metrics.model === state.model.arch ? "chosen" : "");
    for (const value of [names[metrics.model] || "Ingenuo (0 %)", num(metrics.mae_pp, 3),
      num(metrics.rmse_pp, 3), metrics.model === "naive" || metrics.skill_mae == null ? "—" :
        signed(metrics.skill_mae * 100) + " %"]) row.append(node("td", value));
    return row;
  }));
  if (!rows.length) { const row = node("tr"); const cell = node("td", "Sin resultados de rentabilidad disponibles."); cell.colSpan = 4; row.append(cell); $("return-body").append(row); }
}

export function cancelReplay() { replaySerial++; replayRecord = null; }

function stepReplay(delta) {
  const select = $("replay-date"), index = select.selectedIndex + delta;
  if (index < 0 || index >= select.options.length) return;
  select.selectedIndex = index; renderReplay();
}

export async function renderReplay() {
  const token = ++replaySerial, h = access.horizon(), arch = $("replay-model").value;
  const key = arch + "/" + h, previousDate = $("replay-date").value;
  replayRecord = null;
  $("replay-reveal").disabled = true;
  $("replay-outcome").hidden = true;
  $("replay-future-key").hidden = true;
  $("replay-reveal").hidden = false;
  $("replay-reveal").textContent = "Ver qué ocurrió ↗";
  $("replay-badge").textContent = "RESULTADO OCULTO";
  $("replay-status").textContent = "Cargando la previsión registrada…";
  $("replay-chart").style.visibility = "hidden";
  $("replay-metrics").replaceChildren();
  try {
    if (!datesCache.has(key)) datesCache.set(key, (await get("/api/replay/" + key)).origins);
    if (token !== replaySerial) return;
    const origins = datesCache.get(key);
    if (!origins.length) throw Error("No hay previsiones registradas para este plazo.");
    if (replayKey !== key || !$("replay-date").options.length) {
      $("replay-date").replaceChildren(...origins.map(origin => new Option(day(origin), origin)));
      $("replay-date").value = origins.includes(previousDate) ? previousDate : origins[0];
      replayKey = key;
    }
    const origin = $("replay-date").value;
    const record = await get("/api/replay/" + key + "/" + origin);
    if (token !== replaySerial) return;
    replayRecord = record;
    $("replay-heading").textContent = names[arch] + " · " + horizons[h];
    $("replay-status").textContent = num(origins.length) + " orígenes disponibles · objetivo: " + day(record.targetDate);
    $("replay-previous").disabled = $("replay-date").selectedIndex === 0;
    $("replay-next").disabled = $("replay-date").selectedIndex === origins.length - 1;
    for (const [label, value] of [["Precio en el origen", num(record.price, 2)],
      ["Mediana prevista", num(record.median, 2)], ["Límites nominales 80 %", record.interval.map(p => num(p, 2)).join(" – ")]]) {
      const item = node("div"); item.append(node("span", label), node("strong", value), node("small", "USD/barril")); $("replay-metrics").append(item);
    }
    $("replay-provenance").textContent = "Origen: " + day(record.origin) + ". Último ajuste: " + day(record.refitDate) +
      ", con objetivos realizados antes de esa fecha. Objetivo: " + day(record.targetDate) + ".";
    replayChart(record);
    $("replay-chart").style.visibility = "visible";
    $("replay-reveal").disabled = false;
    $("replay-prompt").textContent = "La referencia ingenua mantiene " + num(record.price, 2) +
      " USD/barril. Compara esa referencia con la red y el intervalo antes de descubrir el resultado.";
  } catch (error) { if (token === replaySerial) $("replay-status").textContent = error.message; }
}

async function revealReplay() {
  if (!replayRecord) return;
  const token = replaySerial, record = replayRecord;
  $("replay-reveal").disabled = true;
  try {
    const revealed = await get("/api/replay/" + record.model + "/" + record.horizon + "/" + record.origin + "/reveal");
    if (token !== replaySerial) return;
    replayRecord = revealed;
    replayChart(revealed);
    $("replay-future-key").hidden = false;
    $("replay-reveal").hidden = true;
    $("replay-badge").textContent = "RESULTADO REVELADO";
    const outcome = revealed.outcome, box = $("replay-outcome");
    box.hidden = false;
    box.replaceChildren(node("span", "BRENT OBSERVADO AL VENCIMIENTO", "eyebrow"),
      node("strong", num(outcome.actual, 2) + " USD/bbl", "outcome-price"));
    for (const [label, value] of [["Error de la red", num(outcome.modelError, 2) + " USD/bbl"],
      ["Error del ingenuo", num(outcome.naiveError, 2) + " USD/bbl"],
      ["¿Dentro del intervalo?", outcome.insideInterval ? "Sí" : "No"]]) {
      const item = node("div", null, "outcome-row"); item.append(node("span", label), node("strong", value)); box.append(item);
    }
    const gain = outcome.naiveError - outcome.modelError;
    box.append(node("p", Math.abs(gain) < 0.005 ? "En este origen, ambas previsiones tuvieron un error equivalente." :
      "En este origen, la red tuvo " + num(Math.abs(gain), 2) + " USD/barril " + (gain > 0 ? "menos" : "más") +
      " de error que el ingenuo. Un solo caso no determina la calidad del modelo."));
  } catch (error) {
    if (token !== replaySerial) return;
    $("replay-status").textContent = error.message;
    $("replay-reveal").disabled = false;
  }
}
