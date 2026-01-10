use wasm_bindgen::prelude::*;

mod math;
mod meshing;

#[path = "sim.rs"]
mod simulation;

pub use simulation::{GrayScottParams, Simulation};

#[wasm_bindgen(start)]
pub fn wasm_start() {
  // Keep empty: explicit initialization in JS.
}
