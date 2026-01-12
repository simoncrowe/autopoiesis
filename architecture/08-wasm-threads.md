# architecture/08-wasm-threads.md

## Goal

Replace the current two-worker pipeline (simulation worker + meshing worker) with **one Web Worker hosting a single WASM instance**, and use **WASM threads (Rayon)** for parallel CPU compute *inside* that one instance.

Primary motivations:

- Keep **one** WASM linear memory / heap (no duplicate simulation state).
- Avoid copying large voxel buffers across worker boundaries.
- Preserve a responsive WebGL render loop on the main thread.
- Accept Chromium/Firefox as primary targets (Safari support not required).

Non-goals (initially):

- Removing all workers from the browser (Rayon/WASM threads still use workers under the hood).
- Moving rendering to a worker via `OffscreenCanvas`.

---

## Current state (summary)

- `web/main.js` spawns two workers:
  - simulation worker: owns the simulation, periodically publishes scalar-field snapshots (`v`) into `SharedArrayBuffer`.
  - meshing worker: owns meshing, watches a shared atomic `epoch` and reads `v` snapshots from `SharedArrayBuffer`.
- Benefits today:
  - Simulation and meshing are *coarse-grained concurrent*.
  - Main thread stays mostly free for input + WebGL.
- Cost today:
  - Two separate WASM instances (one per worker) and duplicated WASM memory.
  - Large scalar field data is copied from WASM → SAB every publish.

---

## Proposed architecture (“Plan B”)

### High-level design

- Keep WebGL rendering and UI on the main thread (`web/main.js`).
- Replace two workers with **one** `compute_worker.js`.
- `compute_worker.js`:
  - loads the WASM module once
  - owns simulation state *and* meshing state
  - initializes a Rayon threadpool for parallel compute
  - runs a cooperative loop:
    - step simulation
    - occasionally update keyframes / chunk ranges
    - generate mesh for current camera parameters
    - post mesh buffers back to the main thread

This keeps the only “cross-boundary” payload as **mesh output + stats** (which is unavoidable unless rendering also moves into the worker).

### Data flow

Main thread → compute worker:

- simulation configuration updates (strategy selection, dt, parameters, seeding)
- camera state (position, view radius, iso threshold, shading params)
- runtime knobs (target mesh cadence, snapshot cadence)

Compute worker → main thread:

- mesh buffers (positions, normals, colors, indices)
- timings (sim step time, mesh time)
- counters (steps/sec, total steps)

### What disappears vs today

- No `SharedArrayBuffer` “published snapshot” mechanism between two JS workers.
- No atomic epoch/front-index exchange.
- No copying `sim.v` out into SAB.

Note: you may still keep COOP/COEP headers because WASM threads require `SharedArrayBuffer` anyway. This repo already sets the required headers via `Caddyfile`.

---

## Where Rayon helps in this repo

Rayon is most valuable when used for *data-parallel inner loops*.

### Good first parallelization targets

1) **Gray–Scott `Simulation::step_once`** (`wasm/src/gray_scott.rs`)

- This is a 3D stencil update writing `u2/v2` from `u/v`.
- Parallelization strategy:
  - split by `z` slices or by contiguous linear ranges
  - each thread writes to a disjoint region of `u2/v2`
  - `u` and `v` are read-only during the step
- Determinism: should remain deterministic because the computation is pure and order-independent.

2) **Chunk min/max recomputation** (`recompute_chunk_ranges_from_v`)

- Naturally parallel across chunks.
- Each chunk computes `(min,max)` over its local point range.
- Determinism: deterministic (pure reductions).

3) **Meshing across chunks** (`wasm/src/sim.rs` + `wasm/src/meshing.rs`)

- Current code walks potentially-visible chunks and calls `mesh_region_append(_lerp)`.
- Parallelization strategy:
  - compute the list of visible chunks first (cheap)
  - mesh each chunk into a thread-local `MeshBuffers`
  - merge `MeshBuffers` at the end, fixing indices by offsetting by prior vertex counts
- Determinism: should remain deterministic if the merge order is stable.

### Out of scope: RDME parallelization

**RDME (`wasm/src/rdme.rs`) is explicitly not in scope for parallelization in this plan.**

- The current diffusion/reaction implementation uses RNG and accumulation patterns that are order-dependent.
- Parallelizing it correctly requires a deliberate redesign (per-voxel/counter-based RNG streams and deterministic accumulation), which is a separate project.

For Plan B, RDME remains single-threaded; Rayon/threading work targets Gray–Scott + meshing only.

---

## WASM threads + Rayon: build/runtime requirements

### Browser requirements

- `crossOriginIsolated` true (COOP/COEP headers)
- `SharedArrayBuffer` available

The repo already checks this in `web/main.js` and configures headers in `Caddyfile`.

### Tooling requirements

To use Rayon on `wasm32-unknown-unknown` in the browser, you typically need:

- compiler support for atomics + shared memory in the WASM target
- a WASM threadpool initializer (commonly `wasm-bindgen-rayon`)

Conceptually, the runtime flow inside `compute_worker.js` becomes:

1) `await init()` (wasm-bindgen module init)
2) `await initThreadPool(n)` (spawn workers for Rayon threads)
3) run simulation/meshing work that uses `rayon::join` / `par_iter` / etc.

### Pool sizing

Default guidance:

- `n = max(1, min(available_parallelism, 8))` as a starting point
- expose a debug slider/env knob to tune

Reason: meshing + sim are memory-bandwidth heavy; more threads is not always better.

---

## API / module structure changes (recommended)

To really benefit from “one WASM instance, no extra copies”, avoid JS↔WASM shuttling of large slices.

### Preferred: move orchestration into Rust

Introduce a Rust-side coordinator exported via wasm-bindgen, e.g.:

- `Engine` (exported)
  - owns `Simulation` / `StochasticRdmeSimulation`
  - owns `ScalarFieldMesher`
  - stores camera params
  - exposes:
    - `step_for_budget(ms)` or `step_batch(n)`
    - `publish_keyframe()` (updates mesher keyframes directly from sim output)
    - `generate_mesh()` (uses mesher and returns pointers/lengths)
    - `mesh_*_ptr/len` accessors

This avoids a key pitfall:

- Passing `&[f32]` from JS into WASM via wasm-bindgen almost always implies a copy.
- If Rust calls `mesher.push_keyframe_with_chunk_ranges(&sim.v, ...)` internally, it can do it with zero extra copies.

### Alternative: keep orchestration in JS worker

You can still orchestrate from `compute_worker.js`, but you should add Rust helpers that allow:

- mesher to read directly from simulation buffers without JS-provided slices

In practice, the Rust `Engine` approach tends to be simpler and more “WASM-native”.

---

## Main-thread JS changes (outline)

- Replace `simWorker` + `meshWorker` with a single `computeWorker`.
- Remove SAB allocation (`vSabs`, chunk SABs, control blocks).
- Keep the current pattern of sending camera updates at some cadence.
- Continue receiving `mesh` messages and uploading to WebGL buffers.

Message schema (example):

Main → worker:

- `{ type: "init", dims, seed, simConfig, threadCount? }`
- `{ type: "sim_config", config }`
- `{ type: "camera", pos, radius, iso, color, gradMagGain }`

Worker → main:

- `{ type: "ready" }`
- `{ type: "stats", stepsPerSec, totalSteps }`
- `{ type: "mesh", positions, normals, colors, indices, meshMs, epoch }` (transferable buffers)

---

## Worker loop scheduling

A reasonable first cut is to keep the existing “time-sliced stepping” idea:

- within a `setTimeout(loop, 0)` loop:
  - step the simulation until a time budget is reached
  - periodically publish a keyframe
  - if camera changed or enough time passed, generate a mesh

Because simulation and meshing live in the same worker now, you can also add a simple policy:

- prioritize meshing when the camera moves
- prioritize stepping when camera is idle

If you later want coarse-grained overlap between sim and mesh, you can experiment with Rayon task spawning (`rayon::join`) *within* one frame’s work budget, but be careful about memory contention.

---

## Risks / tradeoffs

- **Oversubscription:** Rayon threads are implemented with workers. Keep pool size reasonable.
- **Meshing parallelization requires a merge step:** must be implemented carefully to avoid blowing up memory and to keep deterministic ordering.
- **RDME parallelization:** explicitly out of scope (RDME remains single-threaded).
- **Main thread still receives mesh data:** large meshes will still be bandwidth-heavy to transfer; consider mesh decimation/LOD and/or limiting mesh cadence.

---

## Implementation plan (incremental)

1) **Unify workers**
   - Add `web/compute_worker.js`.
   - Move the sim + mesher control logic into it.
   - Keep simulation and meshing single-threaded initially.

2) **Move orchestration into WASM**
   - Add an exported `Engine` that owns sim + mesher.
   - Expose mesh pointers/lengths and a `generate_mesh(...)` entrypoint.

3) **Enable WASM threads**
   - Add Rayon + wasm threadpool init.
   - Ensure build flags enable atomics/bulk-memory as required.

4) **Parallelize hotspots**
   - Gray–Scott step
   - chunk min/max recomputation
   - per-chunk meshing

5) **RDME stays single-threaded**
   - explicitly out of scope for parallelization in this plan
   - revisit only as a separate determinism-focused project

---

## Suggested validation

- Add a simple deterministic “hash of state after N steps” debug function in WASM and compare:
  - single-thread vs multi-thread (same seed/params)
  - across multiple runs
- Measure:
  - steps/sec for sim
  - mesh generation ms
  - end-to-end FPS stability
