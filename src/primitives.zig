//! `computeAccel` primitives: domain-agnostic parallel building blocks.
//!
//! Added on demand (docs/node-system-migration.md M3): only algorithms that a
//! concrete workload needed, with a CPU reference and element-wise comparison.

pub const scan = @import("primitives/scan.zig");

pub const Scanner = scan.Scanner;
pub const ScanBinding = scan.Binding;
pub const referenceExclusiveScan = scan.referenceExclusiveScan;
pub const scanBlockCount = scan.blockCount;

test {
    _ = scan;
}
