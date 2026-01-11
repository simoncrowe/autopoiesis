# WebGL 2.0 Proof of Concept

## Canvas

- Create a fullscreen HTML canvas.
- Resize with the window.
- Use WebGL 2.0.

---

## Rendering Pipeline

The PoC MUST:
- upload mesh buffers from wasm
- upload camera matrices as uniforms
- render at least one isosurface mesh
- support transparency blending

---

## Shading

- Basic lighting is sufficient:
  - directional light OR
  - simple lambertian shading

- Vertex color support MUST be implemented if provided.

---

## Interaction

- Basic camera movement:
  - keyboard and/or mouse
- No collision
- No gravity

---

## MVP Acceptance Criteria

- Simulation runs deterministically.
- At least one isosurface mesh is visible.
- Camera can move freely through the scene.
- Mesh color and transparency are configurable.
