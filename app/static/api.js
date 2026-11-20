// Acceso a los datos de la web. Con el servidor local, llama a la API. En la versión
// estática (GitHub Pages, `oilcast export-static`) lee los ficheros de api/ y reconstruye
// las mismas respuestas en el navegador.
export const isStatic =
  document.querySelector('meta[name="estrato-mode"]')?.content === "static";

const files = new Map();

async function readJson(path) {
  if (!files.has(path)) {
    const pending = fetch(path).then((res) => {
      if (!res.ok) throw Error("Faltan datos en la versión publicada.");
      return res.json();
    });
    pending.catch(() => files.delete(path));
    files.set(path, pending);
  }
  return files.get(path);
}

async function replayRows(model, horizon) {
  const data = await readJson("api/replay/" + model + "_" + horizon + ".json");
  return data.rows;
}

// Igual que oilcast.replay.replay_case: el resultado solo se incluye al revelar.
async function replayCase(model, horizon, origin, reveal) {
  const row = (await replayRows(model, horizon)).find((r) => r[0] === origin);
  if (!row) throw Error("No hay una previsión registrada para esa fecha");
  const [, target, refit, price, median, lo, hi, actual] = row;
  const brent = await readJson("api/brent.json");
  const points = (from, to) => {
    const out = [];
    brent.dates.forEach((d, i) => {
      if (d >= from && d <= to) out.push([d, brent.prices[i]]);
    });
    return out;
  };
  const past = points("", origin).slice(-252);
  const result = {
    origin,
    targetDate: target,
    refitDate: refit,
    model,
    horizon: Number(horizon),
    price,
    median,
    interval: [lo, hi],
    history: past,
    revealed: reveal,
  };
  if (reveal) {
    if (actual == null) throw Error("Aún no se conoce el precio de esa fecha");
    result.outcome = {
      actual,
      modelError: Math.abs(median - actual),
      naiveError: Math.abs(price - actual),
      insideInterval: lo <= actual && actual <= hi,
      future: points(origin, target),
    };
  }
  return result;
}

async function staticGet(url, options) {
  if (options?.method && options.method !== "GET")
    throw Error(
      "La versión publicada no actualiza datos ni reentrena. Ejecuta la aplicación en local para hacerlo.",
    );
  const parts = url.replace(/^\/?api\//, "").split("/");
  if (parts[0] === "state") return readJson("api/state.json");
  if (parts[0] === "job") return { status: "idle", message: "", progress: 0 };
  if (parts[0] === "validation")
    return readJson("api/validation/" + parts[1] + "_" + parts[2] + ".json");
  if (parts[0] === "replay") {
    const [, model, horizon, origin, reveal] = parts;
    if (!origin)
      return { origins: (await replayRows(model, horizon)).map((r) => r[0]) };
    return replayCase(model, horizon, origin, reveal === "reveal");
  }
  throw Error("La petición no está disponible en la versión publicada.");
}

export async function get(url, options, fallback = "La petición no se ha podido completar.") {
  if (isStatic) return staticGet(url, options);
  const res = await fetch(url, options);
  if (!res.ok) {
    let body;
    try {
      body = await res.json();
    } catch {}
    throw Error(body?.error || fallback);
  }
  return res.json();
}
