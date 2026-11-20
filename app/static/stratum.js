// Bloque geológico del Brent: la historia diaria como relieve estratificado y la previsión
// como tres testigos de sondeo. Todo se construye con los datos de /api/state.
import * as T from "./vendor/three.js";

const K = 1.18; // escala horizontal: x = ±K·√sesiones (eje comprimido hacia el pasado)
const H = 0.16; // escala vertical: unidades por USD/barril
const D = 16; // fondo del bloque
const NX = 560;
const NZ = 44;
const SMOOTH = 252; // ventana de la media móvil en la cara posterior (sesiones)
const HORIZONS = [5, 63, 252];
const INK = new T.Color("#0d0c0b");
const BONE = new T.Color("#efe9df");
const ACCENT = new T.Color("#e8a33d");

const xPast = (s) => -K * Math.sqrt(Math.max(0, s));
const xFuture = (f) => K * Math.sqrt(Math.max(0, f));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = (t) => t * t * (3 - 2 * t);
const easeOut = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
const money = (v, n = 2) =>
  Number(v).toLocaleString("es-ES", { minimumFractionDigits: n, maximumFractionDigits: n });
const shortDate = (s) =>
  new Intl.DateTimeFormat("es-ES", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })
    .format(new Date(s))
    .replace(" de ", " ");

// --- Material estratificado: franjas de 10 $ con una veta en el precio de hoy ---------------
function strataMaterial(uniforms) {
  const material = new T.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.92,
    metalness: 0,
    side: T.DoubleSide,
  });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vObj;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvObj = position;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vObj;
uniform float uScale;
uniform float uToday;
uniform vec3 uAccent;
float hash1(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
vec3 rock(float band) {
  float h = hash1(band + 3.0);
  vec3 dark = vec3(0.105, 0.092, 0.080);
  vec3 sand = vec3(0.47, 0.39, 0.30);
  vec3 c = mix(dark, sand, 0.15 + 0.85 * h * h);
  return pow(c, vec3(2.2));
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
float price = vObj.y / uScale;
float wob = sin(vObj.x * 0.31 + vObj.z * 0.23) * 0.9 + sin(vObj.x * 1.37 - vObj.z * 0.5) * 0.35;
float lp = price + wob;
float band = floor(lp / 10.0);
float f = fract(lp / 10.0);
vec3 strata = rock(band);
float seam = smoothstep(0.0, 0.035, f) * smoothstep(1.0, 0.965, f);
strata *= mix(0.55, 1.0, seam);
float fine = abs(fract(lp / 2.5) - 0.5);
strata *= mix(0.9, 1.0, smoothstep(0.0, 0.08, fine));
float vein = 1.0 - smoothstep(0.0, 0.42, abs(price - uToday));
diffuseColor.rgb = mix(strata, uAccent, vein * 0.9);`,
      )
      .replace(
        "#include <emissivemap_fragment>",
        `#include <emissivemap_fragment>
totalEmissiveRadiance += uAccent * (1.0 - smoothstep(0.0, 0.3, abs(price - uToday))) * 0.9;`,
      );
  };
  return material;
}

// --- Geometría -----------------------------------------------------------------------------
function prepare(history) {
  const prices = [];
  let last = history.find((r) => r[1] != null)?.[1] ?? 0;
  for (const row of history) {
    if (row[1] != null) last = row[1];
    prices.push(last);
  }
  const n = prices.length;
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + prices[i];
  const mean = (a, b) => {
    const lo = clamp(Math.floor(a), 0, n - 1);
    const hi = clamp(Math.ceil(b), lo + 1, n);
    return (pre[hi] - pre[lo]) / (hi - lo);
  };
  const dates = history.map((r) => r[0]);
  return { n, prices, mean, dates, xMin: xPast(n - 1) };
}

function heightField(data) {
  const { n, mean, xMin } = data;
  const dx = Math.abs(xMin) / (NX - 1);
  const heights = new Float32Array(NX * NZ);
  const xs = new Float32Array(NX);
  for (let ix = 0; ix < NX; ix++) {
    const x = xMin * (1 - ix / (NX - 1));
    xs[ix] = x;
    const s = (x / K) ** 2;
    const span = Math.max(1, ((2 * Math.abs(x)) / (K * K)) * dx);
    const centre = n - 1 - s;
    for (let j = 0; j < NZ; j++) {
      const w = Math.max(span, 1 + SMOOTH * Math.pow(j / (NZ - 1), 1.5));
      heights[j * NX + ix] = mean(centre - w / 2, centre + w / 2 + 1) * H;
    }
  }
  return { xs, heights };
}

function surface(xs, heights) {
  const pos = new Float32Array(NX * NZ * 3);
  for (let j = 0; j < NZ; j++) {
    const z = D / 2 - (j * D) / (NZ - 1);
    for (let ix = 0; ix < NX; ix++) {
      const k = (j * NX + ix) * 3;
      pos[k] = xs[ix];
      pos[k + 1] = heights[j * NX + ix];
      pos[k + 2] = z;
    }
  }
  const index = [];
  for (let j = 0; j < NZ - 1; j++)
    for (let ix = 0; ix < NX - 1; ix++) {
      const a = j * NX + ix,
        b = a + 1,
        c = a + NX,
        d = c + 1;
      index.push(a, c, b, b, c, d);
    }
  const g = new T.BufferGeometry();
  g.setAttribute("position", new T.BufferAttribute(pos, 3));
  g.setIndex(index);
  g.computeVertexNormals();
  return g;
}

function wall(points, normal) {
  // points: [x, z, top] — una tira vertical desde el suelo hasta el relieve
  const pos = [],
    nor = [],
    index = [];
  points.forEach(([x, z, top], i) => {
    pos.push(x, 0, z, x, top, z);
    nor.push(...normal, ...normal);
    if (i) {
      const a = (i - 1) * 2;
      index.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  });
  const g = new T.BufferGeometry();
  g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new T.Float32BufferAttribute(nor, 3));
  g.setIndex(index);
  return g;
}

function ring(radius, y, material, segments = 72) {
  const pts = [];
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pts.push(new T.Vector3(Math.cos(a) * radius, y, Math.sin(a) * radius));
  }
  return new T.LineLoop(new T.BufferGeometry().setFromPoints(pts), material);
}

// --- Escena --------------------------------------------------------------------------------
export function createStratum(host, labelHost) {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let renderer;
  try {
    renderer = new T.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  } catch {
    host.classList.add("stage-off");
    return { setData() {}, setHorizon() {}, setScroll() {}, start() {}, stop() {}, dispose() {} };
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(INK, 1);
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  host.prepend(renderer.domElement);

  const scene = new T.Scene();
  scene.fog = new T.Fog(INK, 70, 210);
  const camera = new T.PerspectiveCamera(34, 1, 0.5, 600);

  scene.add(new T.HemisphereLight(0xf3e7d3, 0x16120f, 1.25));
  const key = new T.DirectionalLight(0xffe2b8, 2.6);
  key.position.set(-40, 60, 50);
  scene.add(key);
  const rim = new T.DirectionalLight(0x9fb3c8, 0.9);
  rim.position.set(60, 25, -60);
  const fill = new T.DirectionalLight(0xffd9a8, 1.3);
  fill.position.set(90, 30, 45);
  scene.add(rim, fill);

  const floor = new T.Mesh(
    new T.PlaneGeometry(900, 500),
    new T.MeshStandardMaterial({ color: 0x110f0e, roughness: 1, metalness: 0 }),
  );
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  const grid = new T.GridHelper(900, 225, 0x2c2723, 0x1d1a17);
  grid.position.y = 0.02;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);

  const uniforms = {
    uScale: { value: H },
    uToday: { value: 0 },
    uAccent: { value: new T.Color().copy(ACCENT).convertSRGBToLinear() },
  };
  const blockMaterial = strataMaterial(uniforms);
  const block = new T.Group();
  const future = new T.Group();
  const story = new T.Group();
  scene.add(block, future, story);

  const labels = [];
  const lineMat = (color, opacity) =>
    new T.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });

  let data,
    cores = {},
    horizon = 63,
    active = false,
    frame = 0,
    scroll = 0,
    built = false,
    introStart = 0,
    width = 1,
    height = 1;
  const pointer = new T.Vector2(),
    pointerSmooth = new T.Vector2();
  const camPos = new T.Vector3(),
    camTarget = new T.Vector3(),
    wantPos = new T.Vector3(),
    wantTarget = new T.Vector3();
  const shots = {};

  function label(text, position, kind, extra = "") {
    const el = document.createElement("div");
    el.className = "stage-label " + kind + (extra ? " " + extra : "");
    if (Array.isArray(text)) text.forEach((t, i) => el.append(Object.assign(document.createElement(i ? "span" : "b"), { textContent: t })));
    else el.textContent = text;
    labelHost.append(el);
    const item = { el, position, kind, visible: 1 };
    labels.push(item);
    return item;
  }

  function clear() {
    for (const group of [block, future, story])
      group.traverse((o) => {
        o.geometry?.dispose();
        if (o.material && o.material !== blockMaterial) o.material.dispose?.();
      });
    block.clear();
    future.clear();
    story.clear();
    labels.splice(0).forEach((l) => l.el.remove());
    cores = {};
  }

  function build(state) {
    clear();
    data = prepare(state.history);
    const today = data.prices[data.n - 1];
    const todayY = today * H;
    uniforms.uToday.value = today;
    const { xs, heights } = heightField(data);
    block.add(new T.Mesh(surface(xs, heights), blockMaterial));
    const row = (j) => Array.from(xs, (x, ix) => [x, D / 2 - (j * D) / (NZ - 1), heights[j * NX + ix]]);
    const column = (ix) =>
      Array.from({ length: NZ }, (_, j) => [xs[ix], D / 2 - (j * D) / (NZ - 1), heights[j * NX + ix]]);
    block.add(new T.Mesh(wall(row(0), [0, 0, 1]), blockMaterial));
    block.add(new T.Mesh(wall(row(NZ - 1), [0, 0, -1]), blockMaterial));
    block.add(new T.Mesh(wall(column(0), [-1, 0, 0]), blockMaterial));
    block.add(new T.Mesh(wall(column(NX - 1), [1, 0, 0]), blockMaterial));

    // Años sobre la base frontal
    const ticks = [];
    let year = Number(data.dates[0].slice(0, 4)) + 1;
    const lastYear = Number(data.dates[data.n - 1].slice(0, 4));
    for (; year <= lastYear; year++) {
      const idx = data.dates.findIndex((d) => d >= year + "-01-01");
      if (idx < 0) continue;
      const x = xPast(data.n - 1 - idx);
      ticks.push(new T.Vector3(x, 0.03, D / 2 + 0.2), new T.Vector3(x, 0.03, D / 2 + 1.4));
      label(String(year), new T.Vector3(x, 0, D / 2 + 2.2), "year");
    }
    block.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(ticks), lineMat(0x8a8072, 0.7)));

    // Hoy: canto del bloque y precio constante hacia el futuro (la referencia ingenua)
    const xEnd = xFuture(252) + 7;
    const flat = new T.LineDashedMaterial({
      color: ACCENT,
      dashSize: 0.6,
      gapSize: 0.5,
      transparent: true,
      opacity: 0.75,
    });
    for (const z of [D / 2, -D / 2]) {
      const line = new T.Line(
        new T.BufferGeometry().setFromPoints([new T.Vector3(0.05, todayY, z), new T.Vector3(xEnd, todayY, z)]),
        flat,
      );
      line.computeLineDistances();
      future.add(line);
    }
    label(["Hoy", money(today) + " $"], new T.Vector3(0, todayY + 1.4, D / 2 + 0.4), "today");
    label("Precio sin cambios", new T.Vector3(xEnd, todayY + 0.9, D / 2 + 3), "flat");

    // Foso del futuro: marco en el suelo
    const pit = [
      [0.3, D / 2],
      [xEnd, D / 2],
      [xEnd, -D / 2],
      [0.3, -D / 2],
    ].map(([x, z]) => new T.Vector3(x, 0.04, z));
    future.add(new T.LineLoop(new T.BufferGeometry().setFromPoints(pit), lineMat(0x5a5148, 0.8)));

    // Testigos de la previsión
    for (const h of HORIZONS) {
      const fc = state.forecasts[h];
      if (!fc) continue;
      cores[h] = core(fc, h);
    }

    // Episodios del catálogo, con su alfiler sobre el frente
    const pins = [];
    state.events
      .filter((e) => e.source === "catalogue")
      .forEach((event, i) => {
        const idx = data.dates.findIndex((d) => d >= event.date);
        if (idx < 0) return;
        const x = xPast(data.n - 1 - idx);
        const y = event.price * H;
        const top = 25.5 + (i % 2) * 2.6;
        pins.push(new T.Vector3(x, y + 0.2, D / 2 + 0.05), new T.Vector3(x, top - 0.4, D / 2 + 0.05));
        label([event.date.slice(0, 4), event.title], new T.Vector3(x, top, D / 2), "event");
      });
    const pinLines = new T.LineSegments(new T.BufferGeometry().setFromPoints(pins), lineMat(0xd9cfbf, 0.6));
    story.add(pinLines);

    // Encuadres de cámara
    const x2008 = xPast(data.n - 1 - data.dates.findIndex((d) => d >= "2008-07-01"));
    const x2020 = xPast(data.n - 1 - data.dates.findIndex((d) => d >= "2020-04-01"));
    shots.hero = (h) => {
      const xc = xFuture(h);
      return {
        pos: new T.Vector3(52 + xc * 0.4, 42, 82),
        target: new T.Vector3(-22 + xc * 0.4, 5, -8),
      };
    };
    shots.story = [
      { pos: new T.Vector3(36, 22, 48), target: new T.Vector3(7, 14, 0) },
      { pos: new T.Vector3(-2, 40, 84), target: new T.Vector3(-26, 11, 0) },
      { pos: new T.Vector3(x2020 + 26, 30, 62), target: new T.Vector3(x2020 - 8, 15, 0) },
      { pos: new T.Vector3(x2008 + 26, 31, 62), target: new T.Vector3(x2008 - 6, 15, 0) },
    ];
    built = true;
  }

  function core(fc, h) {
    const g = new T.Group();
    const x = xFuture(h);
    g.position.set(x, 0, 0);
    const [lo80, hi80] = fc.intervals["80"];
    const [lo50, hi50] = fc.intervals["50"];
    const mats = {
      outer: new T.MeshStandardMaterial({
        color: BONE,
        transparent: true,
        opacity: 0.1,
        roughness: 0.35,
        side: T.DoubleSide,
        depthWrite: false,
      }),
      inner: new T.MeshStandardMaterial({
        color: BONE,
        transparent: true,
        opacity: 0.32,
        roughness: 0.5,
        depthWrite: false,
      }),
      median: new T.MeshStandardMaterial({ color: BONE, roughness: 0.4, emissive: ACCENT, emissiveIntensity: 0 }),
      line: lineMat(BONE, 0.55),
    };
    const r = 1.0;
    const outer = new T.Mesh(new T.CylinderGeometry(r, r, (hi80 - lo80) * H, 64, 1, true), mats.outer);
    outer.position.y = ((lo80 + hi80) / 2) * H;
    const inner = new T.Mesh(new T.CylinderGeometry(r * 0.66, r * 0.66, (hi50 - lo50) * H, 48), mats.inner);
    inner.position.y = ((lo50 + hi50) / 2) * H;
    const median = new T.Mesh(new T.CylinderGeometry(r * 1.12, r * 1.12, 0.12, 64), mats.median);
    median.position.y = fc.price * H;
    const string = new T.Mesh(
      new T.CylinderGeometry(0.05, 0.05, lo80 * H, 8),
      new T.MeshStandardMaterial({ color: 0x6d655b, roughness: 0.8 }),
    );
    string.position.y = (lo80 * H) / 2;
    const collar = ring(r * 1.25, 0.05, mats.line);
    g.add(outer, inner, median, string, collar, ring(r, lo80 * H, mats.line), ring(r, hi80 * H, mats.line));
    future.add(g);
    const names = { 5: "1 semana", 63: "3 meses", 252: "1 año" };
    const tag = label(
      [names[h], money(fc.price) + " $", money(lo80, 1) + " – " + money(hi80, 1)],
      new T.Vector3(x, hi80 * H + 1.6, 0),
      "core",
    );
    tag.el.dataset.horizon = h;
    return { group: g, mats, tag, date: shortDate(fc.date) };
  }

  function paintCores() {
    for (const [h, c] of Object.entries(cores)) {
      const on = Number(h) === horizon;
      c.mats.outer.color.copy(on ? ACCENT : BONE);
      c.mats.outer.opacity = on ? 0.2 : 0.07;
      c.mats.inner.color.copy(on ? ACCENT : BONE);
      c.mats.inner.opacity = on ? 0.55 : 0.22;
      c.mats.median.emissiveIntensity = on ? 1.4 : 0;
      c.mats.median.color.copy(on ? ACCENT : BONE);
      c.mats.line.color.copy(on ? ACCENT : BONE);
      c.mats.line.opacity = on ? 0.9 : 0.4;
      c.tag.el.classList.toggle("on", on);
    }
  }

  function resize() {
    width = host.clientWidth || innerWidth;
    height = host.clientHeight || innerHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    // En pantallas estrechas la cámara se aleja para que quepan los testigos
    camera.fov = width < 700 ? 48 : 34;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  resize();

  const onPointer = (e) => {
    pointer.set((e.clientX / innerWidth) * 2 - 1, (e.clientY / innerHeight) * 2 - 1);
  };
  addEventListener("pointermove", onPointer, { passive: true });

  function desired() {
    const hero = shots.hero(horizon);
    // 0–1: héroe → primer plano de los testigos; 1–3: recorrido hacia el pasado
    const t = clamp(scroll, 0, 2.7);
    let a, b, k;
    if (t <= 1) {
      a = hero;
      b = shots.story[0];
      k = ease(t);
    } else {
      const seg = Math.min(2.999, ((t - 1) / 1.7) * 3);
      const i = Math.floor(seg);
      a = shots.story[i];
      b = shots.story[Math.min(i + 1, 3)];
      k = ease(seg - i);
    }
    wantPos.lerpVectors(a.pos, b.pos, k);
    wantTarget.lerpVectors(a.target, b.target, k);
    const sway = reduced ? 0 : 1;
    wantPos.x += pointerSmooth.x * 2.4 * sway;
    wantPos.y -= pointerSmooth.y * 1.4 * sway;
  }

  function placeLabels() {
    const v = new T.Vector3();
    const storyVis = clamp((scroll - 1.15) / 0.5, 0, 1);
    const coreVis = 1 - clamp((scroll - 1.6) / 0.6, 0, 1);
    let lastYearX = Infinity;
    const placed = [];
    const intro = introStart ? clamp((performance.now() - introStart - 1500) / 700, 0, 1) : 1;
    // Los años se recorren del más reciente al más antiguo para descartar los que se solapan
    for (let i = labels.length - 1; i >= 0; i--) {
      const l = labels[i];
      v.copy(l.position).project(camera);
      let show = v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15;
      const sx = ((v.x + 1) / 2) * width;
      const sy = ((1 - v.y) / 2) * height;
      let opacity = intro;
      if (l.kind === "event") {
        opacity *= storyVis;
        // Si dos episodios se pisan en pantalla, queda el más reciente
        if (show && placed.some(([px, py]) => Math.abs(px - sx) < 172 && Math.abs(py - sy) < 46)) show = false;
        else if (show) placed.push([sx, sy]);
      }
      if (l.kind === "core" || l.kind === "flat") opacity *= coreVis;
      if (l.kind === "year") {
        // En la portada, los años no se cruzan con la columna de texto
        if (scroll < 0.9 && width > 900 && sx < width * 0.46) show = false;
        if (show && Math.abs(lastYearX - sx) < 46) show = false;
        else if (show) lastYearX = sx;
      }
      if (!show) opacity = 0;
      l.el.style.opacity = opacity.toFixed(3);
      l.el.style.transform = `translate3d(${sx.toFixed(1)}px, ${sy.toFixed(1)}px, 0)`;
    }
    story.children.forEach((o) => (o.material.opacity = 0.6 * storyVis));
  }

  function tick(now) {
    if (!active) return;
    frame = requestAnimationFrame(tick);
    if (!built) return;
    pointerSmooth.lerp(pointer, 0.04);
    desired();
    const damping = reduced ? 1 : 0.055;
    camPos.lerp(wantPos, damping);
    camTarget.lerp(wantTarget, damping);
    if (introStart) {
      const t = (now - introStart) / 1000;
      block.scale.y = 0.001 + 0.999 * easeOut(t / 1.8);
      future.children.forEach((o) => {
        if (o.isGroup) o.scale.y = 0.001 + 0.999 * easeOut((t - 1.1) / 1.1);
      });
      if (t > 2.4) introStart = 0;
    }
    camera.position.copy(camPos);
    camera.lookAt(camTarget);
    // Desplaza el encuadre a la derecha para dejar libre la columna de texto
    // En móvil, el texto ocupa la parte alta: la escena baja hacia el pie
    const shift = width > 900 ? 0.1 - 0.02 * clamp(scroll, 0, 1) : 0;
    const drop = width < 700 ? 0.2 * (1 - clamp(scroll, 0, 1)) : 0;
    camera.setViewOffset(width, height, -shift * width, -drop * height, width, height);
    renderer.render(scene, camera);
    placeLabels();
  }

  return {
    setData(state) {
      const first = !built;
      build(state);
      paintCores();
      desired();
      if (first) {
        camTarget.copy(wantTarget);
        camPos.copy(wantPos);
        if (!reduced) {
          camPos.add(new T.Vector3(-30, 46, 40));
          introStart = performance.now();
        }
      }
    },
    setHorizon(h) {
      horizon = h;
      paintCores();
    },
    setScroll(t) {
      scroll = t;
    },
    start() {
      if (active) return;
      active = true;
      resize();
      frame = requestAnimationFrame(tick);
    },
    stop() {
      active = false;
      cancelAnimationFrame(frame);
    },
    dispose() {
      this.stop();
      observer.disconnect();
      removeEventListener("pointermove", onPointer);
      clear();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
