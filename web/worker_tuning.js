export const WORKER_TUNING = Object.freeze({
  // Run the simulation as fast as possible, but only publish snapshots at `periodMs`.
  // Meshing interpolates between the most recent two published snapshots.
  stepTimeSliceMs: 12,
  stepBatch: 8,

  meshIntervalMs: 16,

  // Keep latency low for audio control. This samples a small neighbourhood each interval.
  voxelStatsIntervalMs: 50,
  voxelStatsSide: 6,
});
