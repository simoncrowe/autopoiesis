# abiogenesis

Abiogenesis-inspired pattern exploration MVP:
- 3D Gray–Scott reaction–diffusion simulation (Rust → wasm)
- Isosurface extraction in wasm (marching tetrahedra)
- WebGL 2.0 viewer + free-fly camera (JS)
- Simulation + meshing run in a Web Worker (smooth rendering)

## Build + Run

Prereqs:
- Rust toolchain
- `wasm-pack`

Build wasm:

- `wasm-pack build wasm --release --target web --out-dir web/pkg`

Serve the repo root (any static server works).

Note: this project now uses `SharedArrayBuffer`, so the page must be `crossOriginIsolated` (COOP/COEP headers) and served from a secure context.

- `caddy run`

Open:
- `https://localhost:8080/web/`

## Controls

- Move: `W/A/S/D`
- Look: click canvas for mouse-look
- Speed: mouse wheel
