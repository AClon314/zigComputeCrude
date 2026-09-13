//! M0 runtime: the "resident + chained" layer on top of the WebGPU context.
//!
//! Ablation-first scope (docs/node-system-migration.md):
//!   * `Buffer` — caller-owned device buffer with host/device sync tracking;
//!   * `Kernel` — WGSL + entry point + binding access modes, compiled once;
//!   * `Chain`  — many dispatches, one submit, one readback.
//! Resources pools, views/strides, indirect dispatch and textures are
//! deliberately out of scope until a measured workload asks for them.

pub const buffer = @import("runtime/buffer.zig");
pub const kernel = @import("runtime/kernel.zig");
pub const chain = @import("runtime/chain.zig");

const wgpu = @import("gpu/webgpu.zig");

pub const Buffer = buffer.Buffer;
pub const SyncState = buffer.SyncState;
pub const Kernel = kernel.Kernel;
pub const Binding = kernel.Binding;
pub const Access = kernel.Access;
pub const BindingKind = kernel.BindingKind;
pub const Chain = chain.Chain;

/// The device half of the runtime is still `GpuContext` (instance/adapter/
/// device/queue/limits).  A later milestone can split it without changing
/// callers that only use this alias.
pub const Device = @import("gpu/context.zig").GpuContext;

/// Release a bind group created by `Kernel.createBindGroup`.
pub fn releaseBindGroup(handle: wgpu.WGPUBindGroup) void {
    wgpu.wgpuBindGroupRelease(handle);
}

/// Open the process-local device.  Same context the legacy kernels use, so a
/// program can mix runtime chains and the per-kernel APIs without creating a
/// second instance/adapter/device.
pub fn open() !*Device {
    return @import("gpu/context.zig").global();
}

test {
    _ = buffer;
    _ = kernel;
    _ = chain;
}
