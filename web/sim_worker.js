import init, { Simulation, GrayScottParams } from "../wasm/web/pkg/abiogenesis.js";

let wasm;
let sim;

let camPos = [0, 0, 0.25];
let viewRadius = 0.35;
let iso = 0.5;
let color = [0.15, 0.65, 0.9, 0.9];

const meshIntervalMs = 100; // 10 Hz snapshot mesh
const simBudgetMs = 16; // time slice per worker tick

let lastMeshAt = 0;
let totalSteps = 0;
let stepsWindow = 0;
let lastRateAt = 0;
let stepsPerSec = 0;

function copyFloat32(ptr, len) {
  return new Float32Array(wasm.memory.buffer, ptr, len).slice();
}

function copyU32(ptr, len) {
  return new Uint32Array(wasm.memory.buffer, ptr, len).slice();
}

function postMesh(meshMs) {
  const positions = copyFloat32(sim.mesh_positions_ptr(), sim.mesh_positions_len());
  const normals = copyFloat32(sim.mesh_normals_ptr(), sim.mesh_normals_len());
  const colors = copyFloat32(sim.mesh_colors_ptr(), sim.mesh_colors_len());
  const indices = copyU32(sim.mesh_indices_ptr(), sim.mesh_indices_len());

  const verts = sim.mesh_vertex_count();

  self.postMessage(
    {
      type: "mesh",
      positions: positions.buffer,
      normals: normals.buffer,
      colors: colors.buffer,
      indices: indices.buffer,
      indexCount: indices.length,
      vertexCount: verts,
      meshMs,
      stepsPerSec,
      totalSteps,
    },
    [positions.buffer, normals.buffer, colors.buffer, indices.buffer],
  );
}

function stepSimulationBudgeted() {
  const start = performance.now();
  let localSteps = 0;
  while (performance.now() - start < simBudgetMs) {
    sim.step(1);
    localSteps++;
  }
  totalSteps += localSteps;
  stepsWindow += localSteps;

  const now = performance.now();
  if (lastRateAt === 0) {
    lastRateAt = now;
  }
  const elapsed = now - lastRateAt;
  if (elapsed >= 1000) {
    stepsPerSec = stepsWindow / (elapsed / 1000);
    stepsWindow = 0;
    lastRateAt = now;
  }
}

function maybeMesh() {
  const now = performance.now();
  if (lastMeshAt !== 0 && now - lastMeshAt < meshIntervalMs) {
    return;
  }

  lastMeshAt = now;
  const t0 = performance.now();
  sim.generate_isosurface_mesh_visible(
    camPos[0],
    camPos[1],
    camPos[2],
    viewRadius,
    iso,
    color[0],
    color[1],
    color[2],
    color[3],
  );
  const meshMs = performance.now() - t0;
  postMesh(meshMs);
}

function loop() {
  stepSimulationBudgeted();
  maybeMesh();
  setTimeout(loop, 0);
}

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "camera") {
    if (Array.isArray(msg.pos) && msg.pos.length === 3) {
      camPos = [msg.pos[0], msg.pos[1], msg.pos[2]];
    }
    if (typeof msg.radius === "number") viewRadius = msg.radius;
    if (typeof msg.iso === "number") iso = msg.iso;
    if (Array.isArray(msg.color) && msg.color.length === 4) {
      color = [msg.color[0], msg.color[1], msg.color[2], msg.color[3]];
    }
  }
};

(async () => {
  wasm = await init();

  const params = new GrayScottParams();
  params.set_du(0.16);
  params.set_dv(0.08);
  params.set_feed(0.037);
  params.set_kill(0.06);

  sim = new Simulation(192, 192, 192, 1337, params);
  sim.set_dt(0.1);
  sim.seed_perlin(6.0, 4, 0.0, 1.0);

  // Build the first mesh immediately.
  lastMeshAt = 0;
  lastRateAt = performance.now();

  self.postMessage({ type: "ready" });
  loop();
})().catch((err) => {
  self.postMessage({ type: "error", message: String(err?.stack || err) });
});
