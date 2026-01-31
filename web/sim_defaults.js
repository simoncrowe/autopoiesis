export const GRAY_SCOTT_DEFAULTS = Object.freeze({
  dims: 128,
  dt: 0.1,
  ticksPerSecond: 5,
  params: Object.freeze({
    du: 0.16,
    dv: 0.08,
    feed: 0.0367,
    kill: 0.0649,
  }),
  seeding: Object.freeze({
    type: "perlin",
    frequency: 6.0,
    octaves: 4,
    v_bias: 0.0,
    v_amp: 1.0,
  }),
  exportMode: null,
});

export const STOCHASTIC_RDME_DEFAULTS = Object.freeze({
  dims: 128,
  dt: 0.05,
  ticksPerSecond: 5,
  params: Object.freeze({
    df: 0.2,
    da: 0.05,
    di: 0.02,

    k1: 0.002,
    k2: 0.02,
    k3: 0.001,

    feedBase: 2.0,
    feedNoiseAmp: 0.35,
    feedNoiseScale: 8,

    decayA: 0.01,
    decayI: 0.005,
    decayF: 0.0,

    etaScale: 0.25,
    substeps: 1,

    alivenessAlpha: 0.25,
    alivenessGain: 0.05,
  }),
  seeding: Object.freeze({
    type: "perlin",
    frequency: 6.0,
    octaves: 4,
    baseF: 50,
    baseI: 0,
    aBias: 0.0,
    aAmp: 20.0,
  }),
  exportMode: null,
});

export const CAHN_HILLIARD_DEFAULTS = Object.freeze({
  dims: 128,
  dt: 0.002,
  ticksPerSecond: 5,
  params: Object.freeze({
    a: 1.0,
    kappa: 0.6,
    m: 0.2,
    substeps: 2,
    passMode: 1,
    approxMode: 0,
    phiMean: -0.45,
    noiseAmp: 0.02,
  }),
  seeding: Object.freeze({
    type: "spinodal",
    phiMean: 0.0,
    noiseAmp: 0.02,
  }),
  exportMode: "phase_tanh",
});

export const EXCITABLE_MEDIA_DEFAULTS = Object.freeze({
  dims: 128,
  dt: 0.01,
  ticksPerSecond: 5,
  params: Object.freeze({
    epsilon: 0.03,
    a: 0.85,
    b: 0.01,
    du: 1.0,
    dv: 0.0,
    substeps: 1,
  }),
  seeding: Object.freeze({
    type: "random",
    noiseAmp: 0.02,
    excitedProb: 0.002,
  }),
  exportMode: null,
});

export const REPLICATOR_MUTATOR_DEFAULTS = Object.freeze({
  dims: 128,
  dt: 0.02,
  ticksPerSecond: 5,
  params: Object.freeze({
    types: 4,
    gBase: 0.06,
    gSpread: 0.2,
    dR: 0.03,
    feedRate: 0.04,
    dF: 0.01,
    mu: 0.003,
    diffR: 0.01,
    diffF: 0.2,
    substeps: 2,
  }),
  seeding: Object.freeze({
    type: "uniform",
    noiseAmp: 0.02,
    rBase: 0.012,
    fInit: 0.8,
  }),
  exportMode: null,
});

export const LENIA_DEFAULTS = Object.freeze({
  dims: 64,
  dt: 0.01,
  ticksPerSecond: 5,
  params: Object.freeze({
    radius: 5,
    mu: 0.15,
    sigma: 0.03,
    sharpness: 0.5,
  }),
  seeding: Object.freeze({
    type: "blobs",
    blobCount: 12,
    radius01: 0.08,
    peak: 1.0,
  }),
  exportMode: null,
});

export const SIM_DEFAULTS_BY_STRATEGY_ID = Object.freeze({
  gray_scott: GRAY_SCOTT_DEFAULTS,
  stochastic_rdme: STOCHASTIC_RDME_DEFAULTS,
  cahn_hilliard: CAHN_HILLIARD_DEFAULTS,
  excitable_media: EXCITABLE_MEDIA_DEFAULTS,
  replicator_mutator: REPLICATOR_MUTATOR_DEFAULTS,
  lenia: LENIA_DEFAULTS,
});
