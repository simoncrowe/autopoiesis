# Simulation Engine Requirements

## Core Responsibilities

The simulation engine is responsible for:

- maintaining lattice state
- advancing simulation time
- applying local interaction rules
- exposing lattice-derived scalar data for visualization

The engine must be deterministic given identical initial state and rule set.

---

## World Size

- The logical world size MUST be configurable.
- Initial target configuration:
  - 10,000 x 10,000 x 10,000 cells (logical extent)

Notes:
- The implementation is NOT required to store all cells densely.
- Chunking, sparsity, paging, or other memory strategies are allowed and expected.

---

## Time Model

- Simulation time advances in discrete ticks.
- Each tick applies:
  - a transport / movement phase (optional)
  - a local interaction phase (rule-dependent)

The engine must allow:
- stepping forward one tick
- stepping forward multiple ticks

Backward stepping is NOT required for MVP, but:
- the architecture must not preclude it
- state mutation should be localized and explicit

---

## Determinism

- No randomness without explicit, seedable RNG.
- Given the same:
  - initial lattice state
  - rule system
  - parameters
- the simulation MUST produce identical results.

---

## Engine / Rule Coupling

The simulation engine MUST NOT embed specific chemistry or molecular logic.

Instead:
- the engine calls into a rule system interface
- the rule system decides how local states update

This decoupling is mandatory.
