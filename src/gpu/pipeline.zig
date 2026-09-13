//! Compatibility shim for `add` / `saxpy`.
//!
//! The implementation moved to `primitives/elementwise.zig` (M0 runtime:
//! `runtime.Kernel` + `Buffer` + `Chain`), removing the old per-kernel caches
//! that used to live inside `GpuContext`.  This file keeps the historical
//! module path and API so existing callers (`engine.zig`, `bench.zig`, CLI)
//! keep working.
//!
//! New code should use `computeAccel.runtime` directly for residency/chaining,
//! or `computeAccel.primitives.elementwise` for the per-call API.

const elementwise = @import("../primitives/elementwise.zig");

pub const addWithContext = elementwise.addWithContext;
pub const saxpyWithContext = elementwise.saxpyWithContext;
pub const addBatchedWithContext = elementwise.addBatchedWithContext;

pub const add = elementwise.add;
pub const saxpy = elementwise.saxpy;
pub const addBatched = elementwise.addBatched;

pub const recordFallback = elementwise.recordFallback;
pub const clearFallbackReason = elementwise.clearFallbackReason;
pub const lastFallbackReason = elementwise.lastFallbackReason;
pub const resetCache = elementwise.resetCache;
