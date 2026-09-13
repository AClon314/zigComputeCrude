//! M0 runtime: a compiled compute kernel described by data.
//!
//! A `Kernel` bundles one WGSL module, its entry point, the bind-group layout
//! and the binding access modes.  The access modes are what let `Chain` track
//! which buffers the GPU wrote (and therefore which device copies are newest).
//!
//! Only the WebGPU baseline is assumed: storage/uniform buffers, 64-invocation
//! workgroups, 2D dispatch grids.  Textures, indirect dispatch and push
//! constants are intentionally absent (see docs/node-system-migration.md).

const std = @import("std");
const wgpu = @import("../gpu/webgpu.zig");
const context_mod = @import("../gpu/context.zig");
const GpuContext = context_mod.GpuContext;
const Buffer = @import("buffer.zig").Buffer;

pub const Access = enum { read, write };
pub const BindingKind = enum { storage, uniform };

pub const Binding = struct {
    kind: BindingKind = .storage,
    access: Access = .read,
};

/// WebGPU guarantees at least 8 storage buffers per shader stage; that is the
/// budget a fused chain has to fit into.
pub const max_bindings = 8;

pub const Kernel = struct {
    ctx: *GpuContext,
    slot: context_mod.PipelineCache = .{},
    bindings: []const Binding,
    workgroup_size: u32 = 64,

    /// `entry` must be comptime because the ABI layer builds string views
    /// without allocation.
    pub fn init(
        ctx: *GpuContext,
        shader_code: []const u8,
        comptime entry: []const u8,
        bindings: []const Binding,
        workgroup_size: u32,
    ) !Kernel {
        if (bindings.len == 0 or bindings.len > max_bindings) return error.GpuError;
        if (workgroup_size == 0) return error.GpuError;

        var self = Kernel{
            .ctx = ctx,
            .bindings = bindings,
            .workgroup_size = workgroup_size,
        };
        errdefer self.deinit();

        var entries: [max_bindings]wgpu.WGPUBindGroupLayoutEntry = undefined;
        for (bindings, 0..) |binding, index| {
            const layout_entry = &entries[index];
            layout_entry.* = std.mem.zeroes(wgpu.WGPUBindGroupLayoutEntry);
            layout_entry.binding = @intCast(index);
            layout_entry.visibility = wgpu.WGPUShaderStage_Compute;
            layout_entry.buffer.type = switch (binding.kind) {
                .uniform => wgpu.WGPUBufferBindingType_Uniform,
                .storage => if (binding.access == .read)
                    wgpu.WGPUBufferBindingType_ReadOnlyStorage
                else
                    wgpu.WGPUBufferBindingType_Storage,
            };
        }

        self.slot.bind_group_layout = try ctx.createBindGroupLayout(&wgpu.WGPUBindGroupLayoutDescriptor{
            .nextInChain = null,
            .label = context_mod.emptyStringView(),
            .entryCount = bindings.len,
            .entries = entries[0..].ptr,
        });
        try ctx.createKernelPipeline(&self.slot, shader_code, entry, self.slot.bind_group_layout);
        return self;
    }

    pub fn deinit(self: *Kernel) void {
        self.slot.deinit();
    }

    /// Create a bind group for `buffers`, in binding order.  The caller owns
    /// the handle (release it when the buffers are rebuilt).
    pub fn createBindGroup(self: *const Kernel, buffers: []const *const Buffer) !wgpu.WGPUBindGroup {
        if (buffers.len != self.bindings.len) return error.GpuError;

        var entries: [max_bindings]wgpu.WGPUBindGroupEntry = undefined;
        for (buffers, 0..) |buffer, index| {
            const entry = &entries[index];
            entry.* = std.mem.zeroes(wgpu.WGPUBindGroupEntry);
            entry.binding = @intCast(index);
            entry.buffer = buffer.handle;
            entry.offset = 0;
            entry.size = wgpu.WGPU_WHOLE_SIZE;
        }

        return self.ctx.createBindGroup(&wgpu.WGPUBindGroupDescriptor{
            .nextInChain = null,
            .label = context_mod.emptyStringView(),
            .layout = self.slot.bind_group_layout,
            .entryCount = buffers.len,
            .entries = entries[0..].ptr,
        });
    }

    /// Flattened linear grid for element-wise kernels whose WGSL recovers the
    /// index with `num_workgroups.x * workgroup_size` (see add/saxpy/gemm_simple).
    pub fn gridLinear(self: *const Kernel, elements: usize) !context_mod.WorkgroupGrid {
        const groups = (elements + self.workgroup_size - 1) / self.workgroup_size;
        return self.ctx.limits.workgroupGrid(groups) orelse error.GpuError;
    }
};
