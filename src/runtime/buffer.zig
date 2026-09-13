//! M0 runtime: user-owned, device-resident buffer with sync tracking.
//!
//! This is the first piece of the "resident + chained" runtime described in
//! docs/node-system-migration.md: a caller-owned GPU buffer that knows whether
//! the device or the host holds the newest data, so a chain can skip redundant
//! uploads instead of re-sending the same inputs on every step.
//!
//! Scope is deliberately small (ablation-first): no allocator, no views, no
//! strides.  Those only get added when a measured workload needs them.

const std = @import("std");
const wgpu = @import("../gpu/webgpu.zig");
const context_mod = @import("../gpu/context.zig");
const GpuContext = context_mod.GpuContext;

/// Which side holds the newest bytes.
///
/// - `host`: the device copy is stale/never written; the next `toDevice` must
///   upload.  This is also the state after `init`.
/// - `device`: the GPU holds the newest bytes (e.g. a dispatch wrote this
///   buffer); the next `toDevice` with unchanged host data may be skipped.
/// - `synced`: both sides agree as of the last upload/download.
pub const SyncState = enum { host, device, synced };

/// Common usage combinations (see WGPUBufferUsage).
pub const storage_r = wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst;
pub const storage_rw = wgpu.WGPUBufferUsage_Storage |
    wgpu.WGPUBufferUsage_CopyDst |
    wgpu.WGPUBufferUsage_CopySrc;
pub const uniform = wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst;

pub const Buffer = struct {
    handle: wgpu.WGPUBuffer = null,
    byte_size: usize = 0,
    usage: wgpu.WGPUBufferUsage = 0,
    sync: SyncState = .host,

    /// Lazily created `MapRead | CopyDst` staging buffer used by downloads.
    staging: wgpu.WGPUBuffer = null,

    pub fn init(ctx: *GpuContext, byte_size: usize, usage: wgpu.WGPUBufferUsage) !Buffer {
        if (byte_size == 0) return error.GpuError;
        return .{
            .handle = try ctx.createStorageBuffer(byte_size, usage),
            .byte_size = byte_size,
            .usage = usage,
        };
    }

    pub fn deinit(self: *Buffer, ctx: *GpuContext) void {
        _ = ctx;
        if (self.staging) |handle| wgpu.wgpuBufferRelease(handle);
        self.staging = null;
        if (self.handle) |handle| wgpu.wgpuBufferRelease(handle);
        self.handle = null;
        self.byte_size = 0;
    }

    /// Upload host bytes.  Skipped when the device is already the newest side
    /// (`device`/`synced`), which is what makes a resident buffer cheap to
    /// reuse across chains.  The caller must call `markHostDirty()` after
    /// mutating the host slice, otherwise the skip would upload stale data.
    pub fn toDevice(self: *Buffer, ctx: *GpuContext, bytes: []const u8) !void {
        if (bytes.len != self.byte_size) return error.GpuError;
        if (self.sync != .host) return;
        ctx.writeBytes(self.handle, bytes);
        self.sync = .synced;
    }

    /// Mark the host slice as newer than the device copy (forces the next
    /// `toDevice` to actually upload).
    pub fn markHostDirty(self: *Buffer) void {
        self.sync = .host;
    }

    /// Called by `Chain` after a dispatch that writes this buffer.
    pub fn markDeviceWritten(self: *Buffer) void {
        self.sync = .device;
    }

    /// Blocking readback outside a chain (map + copy + unmap).
    pub fn toHost(self: *Buffer, ctx: *GpuContext, out: []u8) !void {
        if (out.len != self.byte_size) return error.GpuError;
        try self.ensureStaging(ctx);
        try ctx.copyBufferToStaging(self.handle, self.staging, self.byte_size);
        try ctx.readBuffer(self.staging, out);
        self.sync = .synced;
    }

    pub fn ensureStaging(self: *Buffer, ctx: *GpuContext) !void {
        if (self.staging != null) return;
        self.staging = try ctx.createStorageBuffer(
            self.byte_size,
            wgpu.WGPUBufferUsage_MapRead | wgpu.WGPUBufferUsage_CopyDst,
        );
    }
};
