# Isosurface Meshing Requirements

## Purpose

Convert lattice-derived scalar fields into polygonal meshes
for visualization.

---

## Isosurface Extraction

- Use a standard isosurface algorithm:
  - marching cubes OR
  - marching tetrahedra

- Operate on cubic voxel grids.
- Support chunk-based execution.

---

## Multiple Meshes

The system MUST support generating multiple meshes per frame, based on:
- different scalar fields
- different thresholds
- different filtering rules

Meshes must be independent.

---

## Mesh Data

Each mesh MUST provide:
- vertex positions (vec3)
- vertex normals (vec3)
- optional vertex color (vec3 or vec4)

---

## Normals (Soft Shading)

Normals MUST be computed from the scalar field gradient
(e.g. central differences), not face normals.

This is required for organic, non-blocky appearance.

---

## Transparency and Color

- Isosurfaces MUST support alpha transparency.
- Base color MUST be configurable in code.
- Vertex colors derived from scalar values MUST be supported as an option.

---

## Memory Ownership

- Mesh buffers MUST be allocated in wasm memory.
- For each mesh, expose:
  - pointer to vertex buffer
  - vertex count
  - pointer to index buffer
  - index count

JavaScript must not generate geometry.
