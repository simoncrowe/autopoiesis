# Camera and Projection System

## Purpose

Provide a flexible, matrix-based camera for navigating
the generated meshes in clip space.

---

## Camera Model

- Use matrix-based transforms:
  - world matrix
  - view matrix
  - projection matrix

- No gravity, collision, or constraints.

---

## Dimensional Generality

The camera math SHOULD be structured to generalize beyond 3D.

The MVP MUST support:
- 3D -> 2D screen projection

The architecture SHOULD make it feasible to:
- extend transforms to 4D
- experiment with time-as-space representations

This is a design consideration, not an MVP feature.

---

## Camera Location

The camera MAY live in:
- JavaScript, OR
- Rust / wasm

If implemented in wasm:
- matrices must be exposed to JS

---

## Controls

The camera MUST support:
- translation
- rotation

Input handling MAY live in JavaScript.

