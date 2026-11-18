import Chart from "./vendor/chart.js";

const mint = "#83ebc4",
  amber = "#efb671",
  gray = "#8c9ba6";
const fmt = new Intl.NumberFormat("es-ES", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const date = (value, opts) =>
  new Intl.DateTimeFormat("es-ES", { timeZone: "UTC", ...opts }).format(value);
const xy = (day, value) => ({ x: Date.parse(day), y: value });
const trace = (label, data, color, extra = {}) => ({
  label,
  data,
  borderColor: color,
  backgroundColor: color,
  borderWidth: 1.75,
  pointRadius: 0,
  pointHoverRadius: 3,
  ...extra,
});
Chart.defaults.font.family = "Barlow, sans-serif";
Chart.defaults.color = "#acb9bd";
const instances = new Map();
export function removeCharts() {
  instances.forEach((c) => c.destroy());
  instances.clear();
}

function paint(id, datasets, { origin, click, period = 12 } = {}) {
  const canvas = document.getElementById(id);
  instances.get(id)?.destroy();
  const chart = new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      onClick: click,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      layout: {
        padding: { left: 0, right: 7, top: origin ? 15 : 6, bottom: 0 },
      },
      scales: {
        x: {
          type: "linear",
          grid: { display: false },
          border: { color: "#324049" },
          ticks: {
            count: canvas.parentElement.clientWidth < 440 ? 4 : 6,
            maxRotation: 0,
            font: { family: "Plex", size: 9 },
            callback: (value) =>
              date(
                Number(value),
                period > 50
                  ? { year: "numeric" }
                  : { month: "short", year: "2-digit" },
              ).replace(" de ", " "),
          },
        },
        y: {
          border: { display: false },
          grid: { color: "#28353b", drawTicks: false },
          ticks: {
            count: 5,
            padding: 8,
            font: { family: "Plex", size: 9 },
            callback: (v) =>
              Number(v).toLocaleString("es-ES", { maximumFractionDigits: 1 }),
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          displayColors: true,
          backgroundColor: "#1d2c31",
          titleColor: "#eff5f2",
          bodyColor: "#cfdbda",
          borderWidth: 1,
          borderColor: "#486057",
          callbacks: {
            title: (items) =>
              date(items[0].parsed.x, {
                day: "numeric",
                month: "short",
                year: "numeric",
              }),
            label: (item) => {
              const title = item.raw.title || item.dataset.label;
              return title + " · " + fmt.format(item.parsed.y) + " USD";
            },
          },
        },
      },
    },
    plugins: origin
      ? [
          {
            id: "forecast-origin",
            afterDraw: (c) => {
              const {
                  ctx,
                  chartArea: a,
                  scales: { x },
                } = c,
                px = x.getPixelForValue(origin);
              ctx.save();
              ctx.setLineDash([3, 4]);
              ctx.strokeStyle = "#708b8399";
              ctx.beginPath();
              ctx.moveTo(px, a.top);
              ctx.lineTo(px, a.bottom);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.font = "9px Plex, monospace";
              ctx.fillStyle = "#a8b8b2";
              ctx.textAlign = "right";
              ctx.fillText("ÚLTIMO DATO", px - 5, a.top + 7);
              ctx.restore();
            },
          },
        ]
      : [],
  });
  instances.set(id, chart);
  return chart;
}

export function historyChart(state, h, months, ma50, ma200) {
  const last = state.history.at(-1),
    cut = new Date(last[0]);
  cut.setUTCMonth(cut.getUTCMonth() - months);
  const rows = months
    ? state.history.filter((r) => Date.parse(r[0]) >= cut.valueOf())
    : state.history;
  const data = [
    trace(
      "Brent real",
      rows.map((r) => xy(r[0], r[1])),
      mint,
    ),
  ];
  [
    [ma50, 2, "Media 50 d", "#9ca9e0"],
    [ma200, 3, "Media 200 d", "#82949f"],
  ].forEach(([show, index, label, color]) => {
    if (show)
      data.push(
        trace(
          label,
          rows.map((r) => xy(r[0], r[index])),
          color,
          { borderDash: [3, 3], borderWidth: 1 },
        ),
      );
  });
  const fc = Object.entries(state.forecasts)
    .filter(([key]) => Number(key) <= h)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map((p) => p[1]);
  const anchor = xy(last[0], last[1]),
    curve = (fn) => [anchor, ...fc.map((f) => xy(f.date, fn(f)))];
  data.push(
    trace(
      "Límite inferior 80 %",
      curve((f) => f.intervals["80"][0]),
      "#a3835c",
      { borderWidth: 0.75 },
    ),
  );
  data.push(
    trace(
      "Límite superior 80 %",
      curve((f) => f.intervals["80"][1]),
      "#b29568",
      { fill: "-1", backgroundColor: "#d99d5024", borderWidth: 0.75 },
    ),
  );
  data.push(
    trace(
      "Previsión central",
      curve((f) => f.price),
      amber,
      {
        borderDash: [5, 4],
        pointRadius: (c) => (c.dataIndex === fc.length ? 4 : 0),
        pointBackgroundColor: amber,
      },
    ),
  );
  paint("history-chart", data, { origin: anchor.x, period: months || 300 });
}

export function validationChart(rows, name) {
  paint(
    "validation-chart",
    [
      trace(
        "Brent real",
        rows.map((r) => xy(r[0], r[1])),
        mint,
      ),
      trace(
        name,
        rows.map((r) => xy(r[0], r[2])),
        amber,
        { borderWidth: 1.4 },
      ),
      trace(
        "Ingenuo",
        rows.map((r) => xy(r[0], r[3])),
        gray,
        { borderWidth: 1, borderDash: [3, 3] },
      ),
    ],
    { period: 48 },
  );
}

export function cycleChart(state, showAuto, open) {
  const data = [
    trace(
      "Brent real",
      state.history.map((r) => xy(r[0], r[1])),
      mint,
      { borderWidth: 1.2 },
    ),
  ];
  ["peak", "trough"].forEach((kind, i) => {
    const rows = state.events.filter(
      (e) => (showAuto || e.source === "catalogue") && e.kind === kind,
    );
    data.push(
      trace(
        i ? "Caída" : "Pico",
        rows.map((r) => ({
          ...xy(r.date, r.price),
          title: r.title,
          date: r.date,
        })),
        i ? "#a7acff" : amber,
        {
          type: "scatter",
          pointStyle: "triangle",
          pointRotation: i ? 180 : 0,
          pointRadius: 5,
          pointHoverRadius: 7,
          showLine: false,
          order: -1,
        },
      ),
    );
  });
  paint("cycle-chart", data, {
    period: 300,
    click: (evt, _items, chart) => {
      const hit = chart
        .getElementsAtEventForMode(evt, "point", { intersect: true }, false)
        .find((i) => i.datasetIndex > 0);
      if (hit) open(data[hit.datasetIndex].data[hit.index].date);
    },
  });
}
