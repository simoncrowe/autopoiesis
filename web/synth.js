import { SuperSonic } from "https://unpkg.com/supersonic-scsynth@latest/dist/supersonic.js";

const DEFAULT_SYNTHDEF = "sonic-pi-dark_ambience";

const SUPERSONIC_VERSION = "latest";
const CORE_BASE = `https://unpkg.com/supersonic-scsynth-core@${SUPERSONIC_VERSION}/`;
const SYNTHDEFS_BASE = `https://unpkg.com/supersonic-scsynth-synthdefs@${SUPERSONIC_VERSION}/synthdefs/`;
const SAMPLES_BASE = `https://unpkg.com/supersonic-scsynth-samples@${SUPERSONIC_VERSION}/samples/`;

let supersonic = null;
let bootPromise = null;
let nextNodeId = 1000;

const GROUP_SYNTHS = 100;
const GROUP_FX = 101;
const BUS_SYNTH = 20;
const BUS_FX = 22;
const NODE_REVERB = 2000;
const NODE_POST = 2001;

const REVERB_SYNTHDEF = "sonic-pi-fx_reverb";
const POST_SYNTHDEF = "sonic-pi-fx_normaliser";

let fxReady = false;

function ensureInstance() {
  if (supersonic) return supersonic;

  supersonic = new SuperSonic({
    // Core runtime assets (AudioWorklet + WASM)
    workerBaseURL: `${CORE_BASE}workers/`,
    wasmBaseURL: `${CORE_BASE}wasm/`,
    workletUrl: `${CORE_BASE}workers/scsynth_audio_worklet.js`,

    // Synth definition assets (Sonic Pi collection)
    synthdefBaseURL: SYNTHDEFS_BASE,

    // Optional samples (only required by sample-based synthdefs)
    sampleBaseURL: SAMPLES_BASE,

    mode: "postMessage",
  });

  // Recreate persistent node tree on boot/recover.
  supersonic.on("setup", async () => {
    fxReady = false;

    // Groups must exist even if synthdefs aren't loaded yet.
    supersonic.send("/g_new", GROUP_SYNTHS, 0, 0);
    supersonic.send("/g_new", GROUP_FX, 1, GROUP_SYNTHS);
    await supersonic.sync();
  });

  return supersonic;
}

export function getSupersonic() {
  return ensureInstance();
}

export function isAudioBooted() {
  return Boolean(supersonic?.initialized);
}

export async function bootAudio({
  synthdefs = [DEFAULT_SYNTHDEF, REVERB_SYNTHDEF, POST_SYNTHDEF],
} = {}) {
  if (bootPromise) return bootPromise;

  bootPromise = (async () => {
    const s = ensureInstance();
    await s.init();
    await s.loadSynthDefs(synthdefs);
    await s.sync();

    // Now that synthdefs are loaded, we can safely create the FX chain.
    await ensureGlobalReverb();
    return s;
  })();

  try {
    return await bootPromise;
  } catch (e) {
    bootPromise = null;
    throw e;
  }
}

export function noteOn({
  synthdef = DEFAULT_SYNTHDEF,
  note = 60,
  amp = 0.25,
  release = 1.5,
  cutoff = 70,
  pan,
  outBus,
} = {}) {
  const s = ensureInstance();
  if (!s.initialized) throw new Error("audio not booted (call bootAudio() after a user gesture)");

  const nodeId = nextNodeId;
  nextNodeId = (nextNodeId + 1) | 0;

  const resolvedOutBus = Number.isFinite(outBus)
    ? Math.trunc(outBus)
    : (fxReady ? BUS_SYNTH : 0);

  const args = [
    "/s_new",
    synthdef,
    nodeId,
    0,
    GROUP_SYNTHS,
    "out_bus",
    resolvedOutBus,
    "note",
    Math.trunc(note),
    "amp",
    Math.max(0, Math.min(2, Number(amp) || 0)),
    "release",
    Math.max(0.01, Number(release) || 0.01),
    "cutoff",
    Number(cutoff) || 0,
  ];

  if (Number.isFinite(pan)) {
    args.push("pan", Math.max(-1, Math.min(1, Number(pan))));
  }

  s.send(...args);

  return nodeId;
}

export async function ensureGlobalReverb() {
  const s = ensureInstance();
  if (!s.initialized) return false;
  if (!s.loadedSynthDefs?.has?.(REVERB_SYNTHDEF)) return false;

  const hasPost = Boolean(s.loadedSynthDefs?.has?.(POST_SYNTHDEF));

  // Recreate FX chain (idempotent: it overwrites our chosen node ids).
  s.send("/n_free", NODE_REVERB);
  s.send("/n_free", NODE_POST);
  s.send(
    "/s_new",
    REVERB_SYNTHDEF,
    NODE_REVERB,
    0,
    GROUP_FX,
    "in_bus",
    BUS_SYNTH,
    "out_bus",
    hasPost ? BUS_FX : 0,
    "mix",
    0.25,
    "room",
    0.5,
  );

  if (hasPost) {
    s.send(
      "/s_new",
      POST_SYNTHDEF,
      NODE_POST,
      1,
      GROUP_FX,
      "in_bus",
      BUS_FX,
      "out_bus",
      0,
    );
  }
  await s.sync();
  fxReady = true;
  return true;
}

export function setReverb({ mix, room } = {}) {
  if (!fxReady) return;
  const params = {};
  if (Number.isFinite(mix)) params.mix = Math.max(0, Math.min(1, Number(mix)));
  if (Number.isFinite(room)) params.room = Math.max(0, Math.min(1, Number(room)));
  setNode(NODE_REVERB, params);
}

export function getAudioRouting() {
  return {
    groups: { synths: GROUP_SYNTHS, fx: GROUP_FX },
    buses: { synth: BUS_SYNTH, fx: BUS_FX },
    nodes: { reverb: NODE_REVERB, post: NODE_POST },
  };
}

export function setNode(nodeId, params = {}) {
  const s = ensureInstance();
  if (!s.initialized) return;
  if (!Number.isFinite(nodeId)) return;

  const flat = [];
  for (const [k, v] of Object.entries(params)) {
    if (typeof k !== "string" || k.length === 0) continue;
    if (!Number.isFinite(v)) continue;
    flat.push(k, v);
  }
  if (flat.length === 0) return;

  s.send("/n_set", Math.trunc(nodeId), ...flat);
}

export function freeNode(nodeId) {
  const s = ensureInstance();
  if (!s.initialized) return;
  if (!Number.isFinite(nodeId)) return;
  s.send("/n_free", Math.trunc(nodeId));
}
