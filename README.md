# abiogenesis

Abiogenesis-inspired pattern exploration MVP:
- 3D Gray–Scott reaction–diffusion simulation (Rust → wasm)
- Isosurface extraction in wasm (marching tetrahedra)
- WebGL 2.0 viewer + free-fly camera (JS)

## Build + Run

Prereqs:
- Rust toolchain
- `wasm-pack`

Build wasm:

- `wasm-pack build wasm --release --target web --out-dir web/pkg`

Serve the repo root (any static server works):

- `python3 -m http.server 8080`

Open:
- `http://localhost:8080/web/`

## Controls

- Move: `W/A/S/D`
- Look: click canvas for mouse-look
- Speed: mouse wheel
