//! Compile-time bridge for the browser entry; the WGSL bytes remain in add.wgsl.
pub const source = @embedFile("add.wgsl");
