import init, { GrayScottParams, Simulation } from "../wasm/web/pkg/abiogenesis.js";

const CTRL_EPOCH = 0;
const CTRL_FRONT_INDEX = 1;

const TIMING_LAST_PUBLISH_MS = 0;
const TIMING_PERIOD_MS = 1;

const DEFAULT_DIMS = 192;
const DEFAULT_PERIOD_MS = 500;

let wasm;
let sim;

let ctrlI32;
let vViews;
let chunkMinViews;
let chunkMaxViews;

let timingI64;

let chunkMinWasm;
let chunkMaxWasm;

let dims = DEFAULT_DIMS;
let periodMs = DEFAULT_PERIOD_MS;

let currentSeed = 1337;
let simConfig = {
  strategyId: "gray_scott",
  params: { du: 0.16, dv: 0.08, feed: 0.037, kill: 0.06 },
  dt: 0.1,
  seeding: { type: "perlin", frequency: 6.0, octaves: 4, v_bias: 0.0, v_amp: 1.0 },
};

let stepTimer = null;

// Exponential moving average of `Simulation.step(1)` duration.
// Used to start the next step early enough that publishing lands near `periodMs`.
let emaStepMs = 120;

let lastPublishMs = Date.now();

let totalSteps = 0;
let stepsWindow = 0;
let lastRateAt = 0;

function publishSnapshot(nowMs) {
  if (!sim || !ctrlI32 || !vViews || !chunkMinViews || !chunkMaxViews) return;

  const front = Atomics.load(ctrlI32, CTRL_FRONT_INDEX);
  const back = front ^ 1;

  const ptr = sim.v_ptr();
  const len = sim.v_len();
  const v = new Float32Array(wasm.memory.buffer, ptr, len);

  vViews[back].set(v);
  if (chunkMinWasm && chunkMaxWasm) {
    chunkMinViews[back].set(chunkMinWasm);
    chunkMaxViews[back].set(chunkMaxWasm);
  }

  // Publish: flip front, bump epoch.
  Atomics.store(ctrlI32, CTRL_FRONT_INDEX, back);
  Atomics.add(ctrlI32, CTRL_EPOCH, 1);

  lastPublishMs = nowMs;

  if (timingI64) {
    Atomics.store(timingI64, TIMING_LAST_PUBLISH_MS, BigInt(nowMs));
    Atomics.store(timingI64, TIMING_PERIOD_MS, BigInt(periodMs));
  }
}

function scheduleNextStep() {
  const targetPublishMs = lastPublishMs + periodMs;
  const startAtMs = targetPublishMs - emaStepMs;
  const delayMs = Math.max(0, startAtMs - Date.now());
  if (stepTimer) clearTimeout(stepTimer);
  stepTimer = setTimeout(stepOnceAndPublish, delayMs);
}

function stepOnceAndPublish() {
  if (!sim) return;

  const t0 = performance.now();
  sim.step(1);
  const nowMs = Date.now();
  publishSnapshot(nowMs);
  const dtMs = performance.now() - t0;

  totalSteps += 1;
  stepsWindow += 1;
  emaStepMs = emaStepMs * 0.8 + dtMs * 0.2;

  const now = performance.now();
  if (lastRateAt === 0) lastRateAt = now;
  const elapsed = now - lastRateAt;
  if (elapsed >= 1000) {
    const stepsPerSec = stepsWindow / (elapsed / 1000);
    stepsWindow = 0;
    lastRateAt = now;
    self.postMessage({ type: "sim_stats", stepsPerSec, totalSteps });
  }

  scheduleNextStep();
}

function mergeSimConfig(update) {
  if (!update || typeof update !== "object") return;

  if (typeof update.strategyId === "string") simConfig.strategyId = update.strategyId;

  if (typeof update.dt === "number") simConfig.dt = update.dt;

  if (update.params && typeof update.params === "object") {
    simConfig.params = {
      ...simConfig.params,
      ...update.params,
    };
  }

  if (update.seeding && typeof update.seeding === "object") {
    simConfig.seeding = {
      ...simConfig.seeding,
      ...update.seeding,
    };
  }
}

async function ensureWasm() {
  if (!wasm) wasm = await init();
}

async function restartSimulation() {
  await ensureWasm();

  if (stepTimer) {
    clearTimeout(stepTimer);
    stepTimer = null;
  }

  emaStepMs = 120;
  lastRateAt = 0;
  stepsWindow = 0;
  totalSteps = 0;

  if (simConfig.strategyId !== "gray_scott") {
    throw new Error(`unknown simulation strategy: ${String(simConfig.strategyId)}`);
  }

  const params = new GrayScottParams();
  params.set_du(Number(simConfig.params?.du ?? 0.16));
  params.set_dv(Number(simConfig.params?.dv ?? 0.08));
  params.set_feed(Number(simConfig.params?.feed ?? 0.037));
  params.set_kill(Number(simConfig.params?.kill ?? 0.06));

  sim = new Simulation(dims, dims, dims, currentSeed, params);
  sim.set_dt(Number(simConfig.dt ?? 0.1));

  const seeding = simConfig.seeding ?? {};
  if (seeding.type === "perlin") {
    sim.seed_perlin(
      Number(seeding.frequency ?? 6.0),
      Number(seeding.octaves ?? 4),
      Number(seeding.v_bias ?? 0.0),
      Number(seeding.v_amp ?? 1.0),
    );
  } else {
    throw new Error(`unknown seeding type: ${String(seeding.type)}`);
  }

  const chunkLen = sim.chunk_v_len();
  chunkMinWasm = new Float32Array(wasm.memory.buffer, sim.chunk_v_min_ptr(), chunkLen);
  chunkMaxWasm = new Float32Array(wasm.memory.buffer, sim.chunk_v_max_ptr(), chunkLen);

  // Warm start: create two keyframes ASAP so meshing can interpolate.
  sim.step(1);
  publishSnapshot(Date.now());
  sim.step(1);
  publishSnapshot(Date.now());

  self.postMessage({ type: "sim_ready" });
  scheduleNextStep();
}

function applyLiveConfigUpdate(update) {
  if (!sim) return false;
  if (typeof update.dt === "number") {
    sim.set_dt(Number(simConfig.dt));
    return true;
  }
  return false;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "init") {
    const ctrl = msg.ctrl;
    const vSabs = msg.vSabs;
    const chunkMinSabs = msg.chunkMinSabs;
    const chunkMaxSabs = msg.chunkMaxSabs;
    const timing = msg.timing;

    if (Number.isFinite(msg.dims)) dims = Math.trunc(msg.dims);
    if (Number.isFinite(msg.periodMs)) periodMs = Math.trunc(msg.periodMs);

    ctrlI32 = new Int32Array(ctrl);
    vViews = [new Float32Array(vSabs[0]), new Float32Array(vSabs[1])];
    chunkMinViews = [new Float32Array(chunkMinSabs[0]), new Float32Array(chunkMinSabs[1])];
    chunkMaxViews = [new Float32Array(chunkMaxSabs[0]), new Float32Array(chunkMaxSabs[1])];
    timingI64 = timing ? new BigInt64Array(timing) : null;

    const nowMs = Date.now();
    lastPublishMs = nowMs;
    if (timingI64) {
      Atomics.store(timingI64, TIMING_LAST_PUBLISH_MS, BigInt(nowMs));
      Atomics.store(timingI64, TIMING_PERIOD_MS, BigInt(periodMs));
    }

    currentSeed = Number.isFinite(msg.seed) ? (Math.trunc(msg.seed) >>> 0) : 1337;
    if (msg.simConfig) mergeSimConfig(msg.simConfig);

    void restartSimulation().catch((err) => {
      self.postMessage({ type: "error", message: String(err?.stack || err) });
    });
    return;
  }

  if (msg.type === "sim_config") {
    const update = msg.config && typeof msg.config === "object" ? msg.config : msg;

    const prevStrategyId = simConfig.strategyId;
    const prevParams = { ...simConfig.params };
    mergeSimConfig(update);

    const paramsChanged =
      prevParams.du !== simConfig.params.du ||
      prevParams.dv !== simConfig.params.dv ||
      prevParams.feed !== simConfig.params.feed ||
      prevParams.kill !== simConfig.params.kill ||
      update.seeding;

    if (prevStrategyId !== simConfig.strategyId || paramsChanged) {
      void restartSimulation().catch((err) => {
        self.postMessage({ type: "error", message: String(err?.stack || err) });
      });
      return;
    }

    // For now, `dt` is the only supported live update.
    applyLiveConfigUpdate(update);
  }
};

