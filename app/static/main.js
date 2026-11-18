import {
  historyChart,
  validationChart,
  cycleChart,
  removeCharts,
} from "./charts.js";
import { anatomy, palette, createScene } from "./networks.js";

const $ = (id) => document.getElementById(id);
const format = (value, n = 2) =>
  value == null
    ? "—"
    : Number(value).toLocaleString("es-ES", {
        minimumFractionDigits: n,
        maximumFractionDigits: n,
      });
const signed = (value, n = 2) => (value > 0 ? "+" : "") + format(value, n);
const date = (s) =>
  new Intl.DateTimeFormat("es-ES", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(s));
const names = { rnn: "RNN simple", lstm: "LSTM", gru: "GRU", naive: "Ingenuo" };
const icons = {
  database:
    '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  drop: '<path d="M12 2C10 6 5 11 5 15a7 7 0 0 0 14 0c0-4-5-9-7-13Z"/><path d="M8 15a4 4 0 0 0 3 4"/>',
  trend: '<path d="m3 16 6-6 4 4 8-9m-6 0h6v6"/>',
  activity: '<path d="M2 13h5l3-8 4 15 3-9h5"/>',
  range: '<path d="M4 6v12m16-12v12M4 12h16m-12-3-3 3 3 3m8-6 3 3-3 3"/>',
  cube: '<path d="m12 2 9 5v10l-9 5-9-5V7Zm0 10 9-5M3 7l9 5v10"/>',
  arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
  close: '<path d="m6 6 12 12m0-12L6 18"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  reset: '<path d="M3 10a9 9 0 1 1 2 8M3 4v6h6"/>',
  play: '<path d="m7 4 13 8-13 8Z"/>',
  pause: '<path d="M8 4v16m8-16v16"/>',
};
const icon = (key) =>
  '<svg aria-hidden="true" viewBox="0 0 24 24">' + icons[key] + "</svg>";
document
  .querySelectorAll("[data-icon]")
  .forEach((el) => (el.innerHTML = icon(el.dataset.icon)));
const el = (tag, txt, cls) => {
  const node = document.createElement(tag);
  if (txt != null) node.textContent = txt;
  if (cls) node.className = cls;
  return node;
};
function choices(box, options, selected, change) {
  box.replaceChildren(
    ...options.map(([value, label]) => {
      const b = el("button", label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(value === selected));
      b.onclick = () => change(value);
      return b;
    }),
  );
}
async function get(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let body;
    try {
      body = await res.json();
    } catch {}
    throw Error(body?.error || "La petición no se ha podido completar.");
  }
  return res.json();
}
let state,
  h = 63,
  months = 12,
  arch = "rnn",
  validationNet = "rnn",
  current = "",
  preview,
  lab,
  serial = 0,
  timer,
  jobPending = false;
function notice(message) {
  $("notice").hidden = !message;
  $("notice").replaceChildren(document.createTextNode(message || ""));
  if (message) {
    const retry = el("button", "Reintentar");
    retry.onclick = load;
    $("notice").append(" ", retry);
  }
}
function syncHorizons() {
  document.querySelectorAll(".horizon-control").forEach((box) =>
    choices(
      box,
      [
        [5, "5 días"],
        [63, "63 días"],
        [252, "252 días"],
      ],
      h,
      (value) => {
        h = value;
        syncHorizons();
        renderPage(false);
      },
    ),
  );
}
function forecastView() {
  const f = state.forecasts[h],
    last = state.history.at(-1);
  $("spot").textContent = format(last[1]);
  $("pred").textContent = format(f.price);
  $("spot-date").textContent = date(last[0]);
  $("target").textContent = "≈ " + date(f.date);
  $("change").textContent = signed(f.changePct * 100) + " %";
  $("change").parentElement.className =
    "price " + (f.change >= 0 ? "mint" : "amber");
  $("delta").textContent = signed(f.change) + " USD/barril";
  $("interval").textContent = f.intervals["80"]
    .map((v) => format(v, 1))
    .join(" – ");
  $("summary").textContent = f.explanation.summary;
  $("reliability").textContent = f.explanation.reliability;
  $("forecast-list").replaceChildren(
    ...Object.entries(state.forecasts).map(([days, fc]) => {
      const tr = el("tr");
      tr.append(
        ...[
          days + " días",
          date(fc.date),
          format(fc.price),
          signed(fc.changePct * 100) + " %",
          fc.intervals["80"].map((v) => format(v, 1)).join(" – "),
        ].map((v) => el("td", v)),
      );
      return tr;
    }),
  );
  $("risks").replaceChildren(...f.explanation.risks.map((r) => el("li", r)));
  choices(
    $("ranges"),
    [
      [3, "3M"],
      [12, "1A"],
      [60, "5A"],
      [0, "Todo"],
    ],
    months,
    (m) => {
      months = m;
      forecastView();
    },
  );
  historyChart(state, h, months, $("ma50").checked, $("ma200").checked);
  $("mini-title").textContent = state.model.arch_label;
  $("mini-spec").textContent =
    state.model.window + " días de entrada · " + state.features + " variables";
  if (!preview) preview = createScene($("preview-stage"), state.model.arch);
}
function pauseIcon(playing) {
  $("pause").innerHTML = icon(playing ? "pause" : "play");
  $("pause").setAttribute(
    "aria-label",
    playing ? "Pausar simulación" : "Reproducir simulación",
  );
}
function labView(reset) {
  choices(
    $("architectures"),
    Object.keys(anatomy).map((a) => [a, names[a]]),
    arch,
    (a) => {
      arch = a;
      labView(true);
    },
  );
  const info = anatomy[arch];
  $("lab-name").textContent = info.name;
  $("cell-title").textContent = info.title;
  $("cell-copy").textContent = info.copy;
  $("gate-list").replaceChildren(
    ...info.parts.flatMap(([color, title, note]) => {
      const term = el("dt"),
        dot = el("b");
      dot.style.background = palette[color];
      term.append(dot, title);
      return [term, el("dd", note)];
    }),
  );
  const fit = state.architecture.find(
    (r) => r.arch === arch && r.horizon === h,
  );
  $("arch-head").textContent = info.name + " a " + h + " días";
  const metrics = [
    ["Ventana", state.model.window + " días"],
    ["Variables", state.features],
    ["Capas", fit?.num_layers],
    ["Unidades por capa", fit?.hidden_size],
    ["Cuantiles", state.model.quantiles.length],
    ["Parámetros", format(fit?.n_params, 0)],
  ];
  $("arch-stats").replaceChildren(
    ...metrics.map(([key, value]) => {
      const wrap = el("div");
      wrap.append(el("dt", key), el("dd", value ?? "—"));
      return wrap;
    }),
  );
  if (reset || !lab) {
    lab?.dispose();
    lab = createScene($("lab-stage"), arch, true);
    pauseIcon(lab.playing);
    const flat = $("lab-stage").dataset.renderer === "schematic";
    $("scene-help").textContent =
      (flat ? "Proyección 2D · " : "") +
      "Arrastra para girar · rueda para acercar";
  }
}
const columns = [
  ["model", "Modelo", (v) => names[v]],
  ["mae", "MAE", format],
  ["mse", "MSE", format],
  ["rmse", "RMSE", format],
  ["r2", "R²", (v) => format(v, 3)],
  ["skill_mae", "Mejora MAE", (v) => signed(v * 100, 1) + " %"],
  ["skill_rmse", "Mejora RMSE", (v) => signed(v * 100, 1) + " %"],
  ["dm_pvalue", "p DM", (v) => format(v, 3)],
  ["dir_acc", "Dirección", (v) => format(v * 100, 1) + " %"],
  ["coverage80", "Cobertura 80 %", (v) => format(v * 100, 1) + " %"],
  ["n_params", "Parámetros", (v) => format(v, 0)],
  ["train_seconds", "Entrenamiento (s)", (v) => format(v, 2)],
];
const seriesCache = new Map();
async function validationView() {
  const tick = ++serial;
  const header = el("tr");
  header.append(...columns.map((c) => el("th", c[1])));
  $("table-head").replaceChildren(header);
  const rows = state.comparison.filter((r) => r.horizon === h);
  $("table-body").replaceChildren(
    ...rows.map((r) => {
      const row = el("tr");
      if (r.model === state.model.arch) row.className = "chosen";
      row.append(
        ...columns.map(([key, , fn]) =>
          el(
            "td",
            r[key] == null || (r.model === "naive" && key.startsWith("skill_"))
              ? "—"
              : fn(r[key]),
          ),
        ),
      );
      return row;
    }),
  );
  if (!rows.length) {
    $("val-status").textContent = "Aún no hay resultados para este horizonte.";
  }
  const archs = Object.keys(anatomy);
  $("validation-net").replaceChildren(
    ...archs.map((k) => new Option(names[k], k)),
  );
  $("validation-net").value = validationNet;
  $("years").replaceChildren(
    ...state.years
      .filter((r) => r.horizon === h && r.model === validationNet)
      .map((r) => {
        const item = el("div", String(r.year));
        item.append(
          el(
            "strong",
            signed(r.skill_mae * 100, 1) + " %",
            r.skill_mae >= 0 ? "mint" : "amber",
          ),
        );
        return item;
      }),
  );
  $("selection").textContent =
    "Mejora anual del MAE de " +
    names[validationNet] +
    " frente al ingenuo. " +
    state.rule;
  $("method").textContent =
    "El test empieza el " +
    date(state.testStart) +
    ". Cada enero se vuelven a ajustar las redes con los objetivos ya conocidos. Las tres se evalúan sobre los mismos días y con el mismo horizonte. La mediana se recalibra con errores pasados y los intervalos se contrastan con los precios posteriores. Este periodo ya se consultó en rondas previas.";
  const key = validationNet + "/" + h;
  $("val-status").textContent = "Cargando predicciones…";
  $("validation-chart").style.visibility = "hidden";
  try {
    if (!seriesCache.has(key))
      seriesCache.set(key, await get("/api/validation/" + key));
    if (tick !== serial || current !== "validacion") return;
    const data = seriesCache.get(key);
    if (!data.length) throw Error("No hay predicciones realizadas.");
    validationChart(data, names[validationNet]);
    $("validation-chart").style.visibility = "visible";
    $("val-status").textContent =
      format(data.length, 0) +
      " previsiones · fechas objetivo del " +
      date(data[0][0]) +
      " al " +
      date(data.at(-1)[0]);
  } catch (e) {
    if (tick === serial) $("val-status").textContent = e.message;
  }
}
function cycleView() {
  const show = $("auto-events").checked,
    events = state.events
      .filter((e) => show || e.source === "catalogue")
      .reverse();
  $("episodes").replaceChildren(
    ...events.map((e) => {
      const item = el("details", null, "episode");
      item.id = "episode-" + e.date;
      const title = el("summary");
      title.append(
        el("time", date(e.date) + " · " + format(e.price) + " USD/bbl"),
        document.createTextNode(e.title),
      );
      item.append(title, el("p", e.phase + " · " + e.text));
      return item;
    }),
  );
  cycleChart(state, show, (day) => {
    const item = $("episode-" + day);
    if (item) {
      item.open = true;
      item.scrollIntoView({ block: "center" });
      item.querySelector("summary").focus({ preventScroll: true });
    }
  });
}
function renderPage(reset = true) {
  if (!state) return;
  if (current === "observatorio") forecastView();
  if (current === "redes") labView(reset);
  if (current === "validacion") validationView();
  if (current === "ciclos") cycleView();
}
function route() {
  const next = location.hash.slice(1);
  const active = ["redes", "validacion", "ciclos"].includes(next)
    ? next
    : "observatorio";
  if (active === current) return;
  preview?.dispose();
  lab?.dispose();
  preview = null;
  lab = null;
  removeCharts();
  serial++;
  current = active;
  document
    .querySelectorAll(".page")
    .forEach((p) => (p.hidden = p.id !== active));
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.hash === "#" + active) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title =
    "Estrato · " +
    {
      observatorio: "Observatorio del Brent",
      redes: "Redes 3D",
      validacion: "Validación",
      ciclos: "Ciclo histórico",
    }[active];
  renderPage();
  window.scrollTo(0, 0);
}
function dataLabels() {
  $("data-info").textContent =
    "Último Brent publicado en la serie: " +
    date(state.date) +
    ". La descarga genera previsiones con los pesos actuales.";
  $("model-info").textContent =
    state.model.arch_label +
    " · entrenada el " +
    date(state.model.trained_at) +
    " con datos hasta el " +
    date(state.model.last_data_date) +
    ".";
}
async function load() {
  try {
    state = await get("/api/state");
    notice("");
    dataLabels();
    syncHorizons();
    renderPage();
  } catch (e) {
    notice(e.message);
  }
}
function busy(running) {
  for (const button of ["refresh", "retrain", "mode"])
    $(button).disabled = running;
}
async function jobStatus() {
  clearTimeout(timer);
  try {
    const job = await get("/api/job"),
      running = job.status === "running";
    $("job-info").textContent = job.message;
    $("job-progress").hidden = !running;
    $("job-progress").value = job.progress;
    busy(running);
    if (running) {
      jobPending = true;
      timer = setTimeout(jobStatus, 1200);
    } else if (jobPending) {
      jobPending = false;
      if (job.status === "done") {
        seriesCache.clear();
        await load();
      }
    }
  } catch (e) {
    $("job-info").textContent = e.message;
    busy(false);
    if (jobPending) timer = setTimeout(jobStatus, 3500);
  }
}
async function operate(action) {
  busy(true);
  try {
    await get("/api/" + action, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: $("mode").value }),
    });
    jobPending = true;
    await jobStatus();
  } catch (e) {
    $("job-info").textContent = e.message;
    busy(false);
  }
}
$("refresh").onclick = () => operate("refresh");
$("retrain").onclick = () => operate("retrain");
$("open-data").onclick = () => {
  $("data-dialog").showModal();
  jobStatus();
};
$("close-data").onclick = () => $("data-dialog").close();
$("data-dialog").onclick = (e) => {
  if (e.target === $("data-dialog")) {
    const box = $("data-dialog").getBoundingClientRect();
    if (
      e.clientX < box.left ||
      e.clientX > box.right ||
      e.clientY < box.top ||
      e.clientY > box.bottom
    )
      $("data-dialog").close();
  }
};
$("pause").onclick = () => {
  if (lab) pauseIcon(lab.toggle());
};
$("reset").onclick = () => lab?.home();
$("ma50").onchange = $("ma200").onchange = () => {
  if (state) forecastView();
};
$("validation-net").onchange = (e) => {
  validationNet = e.target.value;
  validationView();
};
$("auto-events").onchange = cycleView;
document.querySelector(".skip").onclick = (event) => {
  event.preventDefault();
  $("main").focus();
  $("main").scrollIntoView();
};
window.addEventListener("hashchange", route);
route();
load();
