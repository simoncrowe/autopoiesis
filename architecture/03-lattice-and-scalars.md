# Lattice Representation and Scalar Fields

## Lattice State

Each lattice cell contains a fixed-size state.

The exact encoding is implementation-defined, but must be:
- trivially copyable
- suitable for large-scale storage
- local-update-friendly

Examples:
- particle occupancy bits
- small integer species IDs
- packed flags

---

## Scalar Field Derivation

The engine MUST be able to derive scalar fields from the lattice state.

A scalar field is defined as:

    f(x, y, z) -> float

Scalar fields:
- are derived, not stored
- must not mutate simulation state
- must be deterministic

---

## Required Scalar Fields (MVP)

At least one scalar field MUST be implemented, e.g.:
- total occupancy / density

---

## Optional Scalar Fields

Optional examples:
- per-species density
- interaction activity
- rule-defined "organization" metric

---

## Visualization Separation

Scalar fields are a visualization concern.

They:
- may be filtered or smoothed
- may be interpolated over time
- must not affect simulation correctness
