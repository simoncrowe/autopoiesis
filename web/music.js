function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function pick(rng, arr) {
  return arr[Math.max(0, Math.min(arr.length - 1, Math.floor(rng() * arr.length)))];
}

function createRng(seed = 1234) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}

function chordMaj7(root) {
  return [root, root + 4, root + 7, root + 11];
}

function chordMin7(root) {
  return [root, root + 3, root + 7, root + 10];
}

function chordDom7(root) {
  return [root, root + 4, root + 7, root + 10];
}

// Airier 7th-based voicings (still 4 notes so arps behave well).
function chordMaj7add9(root) {
  // 1 3 7 9 (omit 5)
  return [root, root + 4, root + 11, root + 14];
}

function chordMin7add9(root) {
  // 1 b3 b7 9 (omit 5)
  return [root, root + 3, root + 10, root + 14];
}

function chordDom7add9(root) {
  // 1 3 b7 9 (omit 5)
  return [root, root + 4, root + 10, root + 14];
}

function chordHalfDim7(root) {
  // m7b5: 1 b3 b5 b7
  return [root, root + 3, root + 6, root + 10];
}

// Minimal ambient-ish state machine in C major.
// States yield 7th-based chords (plus a few borrowed/secondary colors);
// transitions are mildly stochastic.
const PROGRESSION = {
  // "_" is a pause/rest.
  _: { chord: null, next: ["I", "IV", "vi", "iii"] },

  I: { chord: (oct) => chordMaj7add9(60 + 12 * oct), next: ["IV", "vi", "ii", "iii", "vii0"] },

  iii: { chord: (oct) => chordMin7add9(64 + 12 * oct), next: ["vi", "IV", "ii", "I"] },
  vi: { chord: (oct) => chordMin7add9(69 + 12 * oct), next: ["ii", "IV", "V/ii", "_"] },
  ii: { chord: (oct) => chordMin7add9(62 + 12 * oct), next: ["V", "V/V", "IV"] },

  IV: { chord: (oct) => chordMaj7add9(65 + 12 * oct), next: ["I", "ii", "V", "iv", "_"] },
  iv: { chord: (oct) => chordMin7(65 + 12 * oct), next: ["I", "bVII", "V"] },
  bVII: { chord: (oct) => chordMaj7add9(58 + 12 * oct), next: ["IV", "I", "V", "_"] },

  // Secondary dominants (gentle forward motion without changing the mood).
  "V/V": { chord: (oct) => chordDom7add9(62 + 12 * oct), next: ["V", "V", "I"] },
  "V/ii": { chord: (oct) => chordDom7add9(57 + 12 * oct), next: ["ii", "ii", "V"] },

  V: { chord: (oct) => chordDom7add9(67 + 12 * oct), next: ["I", "vi", "IV", "_"] },

  vii0: { chord: (oct) => chordHalfDim7(59 + 12 * oct), next: ["I", "iii", "V"] },
};

// Moodier, slower-moving harmony in A minor.
// Intended for long-sustain drones.
export const DRONE_PROGRESSION = {
  _: { chord: null, next: ["i", "VI", "iv"] },

  i: { chord: () => chordMin7(45+12), next: ["VI", "iv", "bVII", "_"] }, // Am7 (A2)
  iv: { chord: () => chordMin7(50+12), next: ["bVII", "V", "i"] }, // Dm7 (D3)
  V: { chord: () => chordDom7(52+12), next: ["i"] }, // E7 (E3)
  VI: { chord: () => chordMaj7(53+12), next: ["iv", "i", "bVII"] }, // Fmaj7 (F3)
  bVII: { chord: () => chordDom7(55+12), next: ["i", "VI", "_"] }, // G7 (G3)
};

export function createMusicEngine({
  bpm = 120,
  seed = 1,
  arpeggiate = false,
  progression = PROGRESSION,
  initialState,
  barsPerChord = 1,
  chordChangeProbability = 0.85,
  retriggerOnHold = true,
  onNote,
  getNoteParams,
} = {}) {
  if (typeof onNote !== "function") throw new Error("createMusicEngine requires onNote(note, params)");

  const rng = createRng(seed);

  let running = false;
  let timer = null;

  const defaultInitialState = progression?.I ? "I" : (progression?.i ? "i" : "_");
  let state = (typeof initialState === "string" && progression?.[initialState]) ? initialState : defaultInitialState;
  let chord = progression?.[state]?.chord ? progression[state].chord(0) : null;
  let arpIdx = 0;
  let stepInBar = 0;
  let barsSinceChord = 0;
  let chordJustChanged = true;

  const secondsPerBeat = 60 / Math.max(1, bpm);
  const stepSeconds = secondsPerBeat / 2; // 8ths
  const stepMs = Math.max(10, Math.round(stepSeconds * 1000));

  const arpOrder = [0, 2, 1, 3];

  function nextChord() {
    const cur = progression?.[state];
    const next = pick(rng, cur?.next || ["_"]);
    state = (progression?.[next]) ? next : defaultInitialState;

    const oct = rng() < 0.15 ? 1 : 0;
    const def = progression?.[state];
    chord = typeof def.chord === "function" ? def.chord(oct) : null;
    arpIdx = 0;
    chordJustChanged = true;
  }

  function tick() {
    if (!running) return;

    if (stepInBar === 0) {
      barsSinceChord++;
      const bpc = Number.isFinite(barsPerChord) ? Math.max(1, Math.trunc(barsPerChord)) : 1;
      if (barsSinceChord >= bpc) {
        if (rng() < Math.max(0, Math.min(1, Number(chordChangeProbability)))) {
          nextChord();
        }
        barsSinceChord = 0;
      }
    }

    if (chord) {
      const paramsBase = typeof getNoteParams === "function" ? (getNoteParams() || {}) : {};

      if (arpeggiate) {
        const chordIdx = arpOrder[arpIdx % arpOrder.length];
        const note = chord[chordIdx];
        arpIdx++;
        onNote(note, { ...paramsBase, chordIndex: chordIdx, chordSize: chord.length });
      } else {
        // No arpeggiation: play the full chord on the bar boundary.
        // Optionally only retrigger when the chord actually changes.
        if (stepInBar === 0 && (retriggerOnHold || chordJustChanged)) {
          for (let i = 0; i < chord.length; i++) {
            onNote(chord[i], { ...paramsBase, chordIndex: i, chordSize: chord.length });
          }
        }
      }
    }

    stepInBar = (stepInBar + 1) % 8; // 8 eighth-notes per bar (4/4)
    chordJustChanged = false;
  }

  return {
    start() {
      if (running) return;
      running = true;
      timer = setInterval(tick, stepMs);
      tick();
    },
    stop() {
      running = false;
      if (timer) clearInterval(timer);
      timer = null;
    },
    get running() {
      return running;
    },
    get stepSeconds() {
      return stepSeconds;
    },
    setBpm(nextBpm) {
      const b = Number(nextBpm);
      if (!Number.isFinite(b) || b <= 0) return;
      bpm = b;
    },
    setArpeggiate(next) {
      arpeggiate = Boolean(next);
    },
  };
}

export function mapMeanToCutoff(mean01) {
  const t = clamp01(mean01);
  // Sonic Pi cutoff is roughly 0..130-ish; keep it musical.
  return 45 + t * 65;
}

export function mapMaxToReverbRoom(max01, volumeThreshold01 = 0.5) {
  // Use the meshing volume threshold (iso) as the scale: when the local max scalar
  // value approaches iso, the camera is near a surface we consider "solid".
  const maxT = clamp01(max01);
  const isoT = clamp01(volumeThreshold01);

  // 1 when max=0 (open space), 0 when max>=iso (near/inside solids).
  const denom = Math.max(1e-6, isoT);
  let room = clamp01((isoT - maxT) / denom);

  // Emphasize the near-surface region.
  room *= room; 
  return 0.5 +  (room * 2) // Scale reverb room size
}
