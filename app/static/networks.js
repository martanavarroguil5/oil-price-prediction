import * as T from "./vendor/three.js";

export const palette = {
  state: "#85ebc6",
  input: "#edb475",
  memory: "#aaa4ff",
  output: "#e3f4eb",
};
export const anatomy = {
  rnn: {
    title: "Estado recurrente",
    name: "RNN simple",
    copy: "Cada día combina las variables de mercado con el estado oculto anterior. El último estado alimenta la predicción de cuantiles.",
    parts: [
      ["input", "Entrada diaria", "Variables del mercado en el paso t."],
      [
        "state",
        "Estado oculto",
        "La célula integra la entrada y transmite su estado al día siguiente.",
      ],
      [
        "output",
        "Salida",
        "El último estado de la ventana llega a la cabeza de cuantiles.",
      ],
    ],
  },
  lstm: {
    title: "Memoria con tres puertas",
    name: "LSTM",
    copy: "Una vía de memoria atraviesa la secuencia. Tres puertas regulan qué información se conserva, qué se incorpora y qué se expone al estado oculto.",
    parts: [
      [
        "memory",
        "Memoria de célula",
        "El carril superior lleva la memoria entre días.",
      ],
      [
        "input",
        "Olvido y entrada",
        "Dos puertas regulan lo que se mantiene y la información nueva que se escribe.",
      ],
      [
        "output",
        "Puerta de salida",
        "Controla qué parte de la memoria se refleja en el estado oculto.",
      ],
    ],
  },
  gru: {
    title: "Actualización selectiva",
    name: "GRU",
    copy: "La GRU conserva un único estado. Dos puertas permiten mezclar información previa con un estado candidato calculado a partir de la observación diaria.",
    parts: [
      [
        "input",
        "Puerta de reinicio",
        "Modula el estado anterior al construir el candidato.",
      ],
      [
        "memory",
        "Puerta de actualización",
        "Pondera el estado anterior y el candidato.",
      ],
      ["state", "Estado oculto", "La mezcla resultante pasa al siguiente día."],
    ],
  },
};

const vec = (x, y, z) => new T.Vector3(x, y, z);

// Each slab is the same cell at a successive time step. Interior nodes are representative.
function cellGeometry(arch) {
  const nodes = [],
    edges = [],
    flows = [];
  function node(p, color, radius) {
    nodes.push({ p, color, radius });
    return p;
  }
  function edge(a, b, color, strength = 0.35) {
    edges.push({ a, b, color, strength });
  }
  function route(points, color) {
    const curve = new T.CatmullRomCurve3(points, false, "centripetal");
    const samples = curve.getPoints(30);
    for (let i = 1; i < samples.length; i++)
      edge(samples[i - 1], samples[i], color, 0.88);
    flows.push({ curve, color });
  }
  const steps = [-2.55, 0, 2.55];
  for (const x of steps) {
    const grid = [];
    const cols = 5,
      rows = 5;
    for (let j = 0; j < rows; j++) {
      grid[j] = [];
      for (let k = 0; k < cols; k++) {
        const p = node(
          vec(x, (j - 2) * 0.4, (k - 2) * 0.42),
          palette.state,
          0.055,
        );
        grid[j][k] = p;
        if (j) edge(grid[j - 1][k], p, palette.state, 0.54);
        if (k) edge(grid[j][k - 1], p, palette.state, 0.45);
        if (j && k) edge(grid[j - 1][k - 1], p, palette.state, 0.19);
      }
    }
    const inlet = node(vec(x - 0.38, -1.36, 0.67), palette.input, 0.105);
    route([inlet, vec(x - 0.25, -1.05, 0.67), grid[0][3]], palette.input);
    if (arch !== "rnn") {
      const gates =
        arch === "lstm"
          ? [palette.input, palette.input, palette.output]
          : [palette.input, palette.memory];
      gates.forEach((color, i) => {
        const y = (i - (gates.length - 1) / 2) * 0.6;
        const p = node(vec(x - 0.48, y, 0.98), color, 0.115);
        edge(inlet, p, color, 0.65);
        edge(p, grid[Math.round(y / 0.4) + 2][3], color, 0.8);
      });
    }
    if (arch === "lstm") {
      const m = node(vec(x, 1.42, -0.15), palette.memory, 0.115);
      route([grid[4][2], vec(x, 1.1, 0.05), m], palette.memory);
    }
  }
  steps.slice(1).forEach((x, i) => {
    const a = steps[i];
    for (let j = -2; j <= 2; j++) {
      for (let k = -2; k <= 2; k += 2) {
        edge(
          vec(a, j * 0.4, k * 0.42),
          vec(x, (((j + k + 7) % 5) - 2) * 0.4, (((k + j + 7) % 5) - 2) * 0.42),
          palette.state,
          0.28,
        );
      }
    }
    route(
      [
        vec(a, 0, 0.84),
        vec(a + 0.65, 0.08, 0.92),
        vec(x - 0.64, -0.06, 0.95),
        vec(x, 0, 0.84),
      ],
      palette.state,
    );
    if (arch === "lstm")
      route(
        [
          vec(a, 1.42, -0.15),
          vec((a + x) / 2, 1.42, -0.15),
          vec(x, 1.42, -0.15),
        ],
        palette.memory,
      );
    if (arch === "gru")
      route(
        [
          vec(a, 0.4, -0.42),
          vec(a + 0.55, 1.24, -0.42),
          vec(x - 0.55, 1.24, -0.42),
          vec(x - 0.48, 0.3, 0.98),
        ],
        palette.memory,
      );
  });
  const final = steps.at(-1);
  const head = node(vec(final + 1.1, 0.48, 0), palette.output, 0.13);
  route(
    [vec(final, 0, 0.84), vec(final + 0.55, 0.16, 0.48), head],
    palette.output,
  );
  return { nodes, edges, flows };
}

export function createScene(container, arch, interactive = false) {
  const model = cellGeometry(arch);
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(32, 1, 0.1, 150);
  const group = new T.Group();
  scene.add(group);
  const center = vec(0.35, 0.06, 0);
  let renderer, context2d;
  const canvas = document.createElement("canvas");
  container.replaceChildren(canvas);
  try {
    renderer = new T.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "low-power",
    });
  } catch {
    // A failed WebGL context creation can prevent a 2D context on the same canvas.
    const flat = document.createElement("canvas");
    container.replaceChildren(flat);
    context2d = flat.getContext("2d");
  }
  const surface = container.firstChild;
  container.dataset.renderer = renderer ? "webgl" : "schematic";
  let controls,
    width = 1,
    height = 1,
    yaw = 0,
    pitch = 0,
    scale = 1,
    dragging = null;
  let frame = 0,
    elapsed = 0,
    last = 0,
    visible = false,
    disposed = false,
    firstFrame = true;
  let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pulses = [];
  if (renderer) {
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    const sphere = new T.SphereGeometry(1, 12, 8);
    const material = new T.MeshStandardMaterial({
      roughness: 0.26,
      metalness: 0.2,
    });
    const mesh = new T.InstancedMesh(sphere, material, model.nodes.length);
    const matrix = new T.Object3D();
    model.nodes.forEach(({ p, color, radius }, i) => {
      matrix.position.copy(p);
      matrix.scale.setScalar(radius);
      matrix.updateMatrix();
      mesh.setMatrixAt(i, matrix.matrix);
      mesh.setColorAt(i, new T.Color(color));
    });
    group.add(mesh);
    const positions = [],
      colors = [];
    model.edges.forEach(({ a, b, color, strength }) => {
      positions.push(...a.toArray(), ...b.toArray());
      const c = new T.Color(color).multiplyScalar(strength);
      colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
    });
    const geometry = new T.BufferGeometry();
    geometry.setAttribute(
      "position",
      new T.Float32BufferAttribute(positions, 3),
    );
    geometry.setAttribute("color", new T.Float32BufferAttribute(colors, 3));
    group.add(
      new T.LineSegments(
        geometry,
        new T.LineBasicMaterial({ vertexColors: true }),
      ),
    );
    model.flows.forEach(({ color }) => {
      const m = new T.Mesh(sphere, new T.MeshBasicMaterial({ color }));
      m.scale.setScalar(0.065);
      group.add(m);
      pulses.push(m);
    });
    scene.add(new T.AmbientLight(0xd6fff2, 1.6));
    const key = new T.DirectionalLight(0xeefff4, 2.8);
    key.position.set(3, 6, 5);
    scene.add(key);
    if (interactive) {
      controls = new T.OrbitControls(camera, canvas);
      controls.enablePan = false;
      controls.target.copy(center);
      controls.minDistance = 4;
      controls.maxDistance = 24;
      controls.addEventListener("change", request);
    }
  }
  function home() {
    yaw = 0;
    pitch = 0;
    scale = 1;
    const distance = Math.max(8.0, 14.2 / Math.max(width / height, 0.45));
    camera.position
      .copy(center)
      .add(vec(4.3, 2.7, 8.8).normalize().multiplyScalar(distance));
    camera.lookAt(center);
    controls?.update();
    request();
  }
  function drawFlat() {
    camera.updateMatrixWorld();
    group.updateMatrixWorld();
    const ctx = context2d;
    const project = (p) => {
      const v = p.clone().applyMatrix4(group.matrixWorld).project(camera);
      return [((v.x + 1) * width) / 2, ((1 - v.y) * height) / 2];
    };
    ctx.clearRect(0, 0, width, height);
    model.edges.forEach(({ a, b, color, strength }) => {
      ctx.globalAlpha = strength;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(...project(a));
      ctx.lineTo(...project(b));
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
    const dot = (p, color, r) => {
      const xy = project(p);
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(...xy, r, 0, Math.PI * 2);
      ctx.fill();
    };
    model.nodes.forEach(({ p, color, radius }) =>
      dot(p, color, Math.max(1.5, radius * height * 0.17)),
    );
    model.flows.forEach(({ curve, color }, i) =>
      dot(curve.getPointAt((elapsed * 0.27 + i * 0.17) % 1), color, 2),
    );
  }
  function render(now) {
    frame = 0;
    if (disposed || (!visible && !firstFrame) || document.hidden) return;
    firstFrame = false;
    if (playing && last) elapsed += Math.min((now - last) / 1000, 0.06);
    last = now;
    group.rotation.set(pitch, yaw, 0);
    group.scale.setScalar(scale);
    if (renderer) {
      model.flows.forEach(({ curve }, i) =>
        pulses[i].position.copy(
          curve.getPointAt((elapsed * 0.27 + i * 0.17) % 1),
        ),
      );
      renderer.render(scene, camera);
    } else drawFlat();
    if (playing) request();
  }
  function request() {
    if (!frame && !disposed && (visible || firstFrame) && !document.hidden)
      frame = requestAnimationFrame(render);
  }
  function resize() {
    width = container.clientWidth;
    height = container.clientHeight;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    if (renderer) renderer.setSize(width, height, false);
    else {
      const d = Math.min(devicePixelRatio, 2);
      surface.width = width * d;
      surface.height = height * d;
      context2d.setTransform(d, 0, 0, d, 0, 0);
    }
    firstFrame = true;
    home();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  const io = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    last = 0;
    request();
  });
  io.observe(container);
  const keyboard = (e) => {
    const moves = {
      ArrowLeft: [-0.14, 0],
      ArrowRight: [0.14, 0],
      ArrowUp: [0, -0.14],
      ArrowDown: [0, 0.14],
    };
    if (moves[e.key]) {
      e.preventDefault();
      yaw += moves[e.key][0];
      pitch += moves[e.key][1];
      request();
    }
  };
  const down = (e) => {
    dragging = [e.clientX, e.clientY];
    surface.setPointerCapture(e.pointerId);
  };
  const move = (e) => {
    if (dragging) {
      yaw += (e.clientX - dragging[0]) * 0.01;
      pitch += (e.clientY - dragging[1]) * 0.01;
      dragging = [e.clientX, e.clientY];
      request();
    }
  };
  const up = () => {
    dragging = null;
  };
  const zoom = (e) => {
    e.preventDefault();
    scale = Math.max(0.65, Math.min(1.8, scale * Math.exp(-e.deltaY * 0.001)));
    request();
  };
  if (interactive) {
    container.addEventListener("keydown", keyboard);
    if (!renderer) {
      surface.onpointerdown = down;
      surface.onpointermove = move;
      surface.onpointerup = surface.onpointercancel = up;
      surface.addEventListener("wheel", zoom, { passive: false });
    }
  }
  document.addEventListener("visibilitychange", request);
  resize();
  return {
    playing,
    home,
    toggle() {
      playing = !playing;
      last = 0;
      request();
      return playing;
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      ro.disconnect();
      io.disconnect();
      controls?.dispose();
      document.removeEventListener("visibilitychange", request);
      container.removeEventListener("keydown", keyboard);
      surface.removeEventListener("wheel", zoom);
      const resources = new Set();
      group.traverse((o) => {
        if (o.geometry) resources.add(o.geometry);
        if (o.material) resources.add(o.material);
        if (o.isInstancedMesh) o.dispose();
      });
      resources.forEach((r) => r.dispose());
      renderer?.dispose();
      renderer?.forceContextLoss();
      container.replaceChildren();
    },
  };
}
