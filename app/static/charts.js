import Chart from "./vendor/chart.js";
import { scatterData } from "./scatter-stats.js";

const mint = "#efe9df",
  amber = "#e8a33d",
  gray = "#93a3b1";
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
Chart.defaults.color = "#b5ac9f";
const instances = new Map();
export function removeCharts() {
  instances.forEach((c) => c.destroy());
  instances.clear();
}

function paint(id, datasets, { origin, originLabel = "Último dato", click, period = 12 } = {}) {
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
          border: { color: "#34302b" },
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
          grid: { color: "#25221e", drawTicks: false },
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
          backgroundColor: "#1b1916",
          titleColor: "#efe9df",
          bodyColor: "#b5ac9f",
          borderWidth: 1,
          borderColor: "#3a352f",
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
              ctx.strokeStyle = "#8a807299";
              ctx.beginPath();
              ctx.moveTo(px, a.top);
              ctx.lineTo(px, a.bottom);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.font = "9px Plex, monospace";
              ctx.fillStyle = "#b5ac9f";
              ctx.textAlign = "right";
              ctx.fillText(originLabel, px - 5, a.top + 7);
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
    [ma50, 2, "Media 50 d", "#c4a978"],
    [ma200, 3, "Media 200 d", "#7c8790"],
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
  const fc = state.forecasts[h], anchor = xy(last[0], last[1]);
  data.push(...forecastPoints(fc.date, fc.price, fc.intervals["80"], last));
  paint("history-chart", data, { origin: anchor.x, period: months || 300 });
}


function forecastPoints(target, median, interval, last) {
  return [
    trace("Precio sin cambios", [xy(last[0], last[1]), xy(target, last[1])], gray,
      { borderDash: [3, 5], borderWidth: 1 }),
    trace("Límites nominales 80 % al vencimiento", interval.map(p => xy(target, p)), amber,
      { borderWidth: 3, pointRadius: 3, pointStyle: "rectRot" }),
    trace("Mediana al vencimiento", [xy(target, median)], amber,
      { type: "scatter", pointRadius: 6, pointHoverRadius: 8, showLine: false }),
  ];
}
export function replayChart(record) {
  const data = [trace("Brent hasta el origen", record.history.map(r => xy(r[0], r[1])), mint)];
  data.push(...forecastPoints(record.targetDate, record.median, record.interval,
    [record.origin, record.price]));
  if (record.outcome) data.push(trace("Brent observado después", record.outcome.future.map(r =>
    xy(r[0], r[1])), "#93a3b1", { borderWidth: 2 }));
  paint("replay-chart", data, { origin: Date.parse(record.origin), originLabel: "Origen", period: 24 });
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

export function scatterChart(rows, name, unit, compare) {
  const result = scatterData(rows, unit);
  const values = [...result.points.flatMap(p=>[p.x,p.y]), ...(compare ? result.naive.map(p=>p.y) : [])];
  const low = Math.min(...values), high = Math.max(...values), pad = Math.max((high-low)*.06, .5);
  const min = low-pad, max = high+pad, label = unit === "retorno" ? "p. p." : "USD/barril";
  const sets = [{label:name,data:result.points,backgroundColor:"#e8a33d99",pointRadius:2.2,pointHoverRadius:5,order:1},
    {type:"line",label:"Predicción perfecta · y = x",data:[{x:min,y:min},{x:max,y:max}],
      borderColor:"#efe9df99",borderDash:[5,5],borderWidth:1.2,pointRadius:0,order:0}];
  if (compare) sets.push({label:"Ingenuo",data:result.naive,backgroundColor:"#93a3b14d",pointRadius:2,pointHoverRadius:4,order:2});
  instances.get("scatter-chart")?.destroy();
  const axis = title => ({type:"linear",min,max,title:{display:true,text:title+" ("+label+")",color:"#b5ac9f",font:{size:11}},
    grid:{color:"#25221e"},border:{color:"#34302b"},ticks:{maxTicksLimit:6,font:{family:"Plex",size:9},
      callback:v=>Number(v).toLocaleString("es-ES",{maximumFractionDigits:1})}});
  const chart = new Chart(document.getElementById("scatter-chart"),{type:"scatter",data:{datasets:sets},options:{
    responsive:true,maintainAspectRatio:false,animation:false,parsing:false,
    scales:{x:axis("Real observado"),y:axis("Predicción")},
    plugins:{legend:{display:false},tooltip:{backgroundColor:"#1b1916",callbacks:{
      title:items=>items[0].raw.date ? date(Date.parse(items[0].raw.date),{day:"numeric",month:"short",year:"numeric"}) : "Referencia ideal",
      label:item=>item.dataset.label+": real "+fmt.format(item.parsed.x)+" · previsto "+fmt.format(item.parsed.y)+" "+label,
    }}},
  }});
  instances.set("scatter-chart",chart);
  return result;
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
        i ? "#93a3b1" : amber,
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
