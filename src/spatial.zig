//! `computeAccel_spatial` — domain-agnostic spatial primitives (S1).
//!
//! Scope rule (docs/node-system-migration.md §6): only algorithms/data
//! structures whose semantics do not mention a specific consumer (Blender,
//! physics, rendering, collision) live here, and they are added on demand.
//! No Blender domain model, no node semantics.
//!
//! Current content:
//!   * `grid_hash` — uniform-grid spatial index: build + radius query, with a
//!     CPU reference implementation.  The GPU kernels (atomics + scan +
//!     scatter) land in the next step and are validated against this reference.

pub const grid_hash = @import("spatial/grid_hash.zig");

pub const Grid = grid_hash.Grid;
pub const Vec3 = grid_hash.Vec3;
pub const build = grid_hash.build;
pub const queryCounts = grid_hash.queryCounts;
pub const bruteForceCounts = grid_hash.bruteForceCounts;
pub const gridCovering = grid_hash.gridCovering;

test {
    _ = grid_hash;
}
