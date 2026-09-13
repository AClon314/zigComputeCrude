const std = @import("std");
const wgpu = @import("webgpu.zig");
const context_mod = @import("context.zig");
const GpuContext = context_mod.GpuContext;

const add_shader = @embedFile("shaders/add.wgsl");
const saxpy_shader = @embedFile("shaders/saxpy.wgsl");

const Kernel = enum { add, saxpy };

const WorkgroupGrid = struct {
    x: u32,
    y: u32,
};

fn workgroupCount(n: usize) usize {
    return (n + 63) / 64;
}

/// WebGPU limits each dispatch axis independently.  Flattening the linear
/// workgroup stream into a 2D grid keeps both axes within the device limit and
/// lets the WGSL kernel recover the same linear element index from
/// `num_workgroups`; unlike repeated dispatches, this needs no offset uniform.
fn dispatchGrid(self: *const GpuContext, groups: usize) !WorkgroupGrid {
    if (groups == 0) return error.GpuError;
    const max_dimension: u64 = self.limits.maxComputeWorkgroupsPerDimension;
    if (max_dimension == 0) return error.GpuError;

    const group_count: u64 = @intCast(groups);
    const x = @min(group_count, max_dimension);
    const y = (group_count - 1) / x + 1;
    if (y > max_dimension) return error.GpuError;

    return .{
        .x = @intCast(x),
        .y = @intCast(y),
    };
}

fn kernelIndex(kernel: Kernel) usize {
    return switch (kernel) {
        .add => 0,
        .saxpy => 1,
    };
}

fn emptyStringView() wgpu.WGPUStringView {
    return .{ .data = null, .length = wgpu.WGPU_STRLEN };
}

fn stringView(comptime text: []const u8) wgpu.WGPUStringView {
    return .{ .data = text.ptr, .length = text.len };
}

fn layoutEntries(kernel: Kernel) [4]wgpu.WGPUBindGroupLayoutEntry {
    var entries: [4]wgpu.WGPUBindGroupLayoutEntry = undefined;
    for (&entries) |*entry| entry.* = std.mem.zeroes(wgpu.WGPUBindGroupLayoutEntry);

    const storage_count: usize = 3;
    for (0..storage_count) |binding| {
        entries[binding].binding = @intCast(binding);
        entries[binding].visibility = wgpu.WGPUShaderStage_Compute;
        entries[binding].buffer.type = if (binding == 2)
            wgpu.WGPUBufferBindingType_Storage
        else
            wgpu.WGPUBufferBindingType_ReadOnlyStorage;
    }
    if (kernel == .saxpy) {
        entries[3].binding = 3;
        entries[3].visibility = wgpu.WGPUShaderStage_Compute;
        entries[3].buffer.type = wgpu.WGPUBufferBindingType_Uniform;
    }
    return entries;
}

fn ensurePipeline(self: *GpuContext, kernel: Kernel) !void {
    const cache = &self.pipelines[kernelIndex(kernel)];
    if (cache.pipeline != null) return;

    // This slot owns its bind group layout; the shared pipeline helper only
    // fills in shader module / pipeline layout / pipeline.
    var entries = layoutEntries(kernel);
    cache.bind_group_layout = try self.createBindGroupLayout(&wgpu.WGPUBindGroupLayoutDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .entryCount = if (kernel == .add) 3 else 4,
        .entries = entries[0..].ptr,
    });

    const shader_code = if (kernel == .add) add_shader else saxpy_shader;
    try self.createKernelPipeline(cache, shader_code, "main", cache.bind_group_layout);
}

fn ensureResources(self: *GpuContext, kernel: Kernel, byte_size: usize) !void {
    const index = kernelIndex(kernel);
    const resources = &self.resources[index];
    if (resources.byte_size == byte_size and resources.bind_group != null) return;

    resources.deinit();
    resources.byte_size = byte_size;
    errdefer resources.deinit();

    for (0..3) |i| {
        const usage = if (i == 2)
            wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc
        else
            wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst;
        resources.storage[i] = try self.createStorageBuffer(byte_size, usage);
    }

    if (kernel == .saxpy) {
        // A one-f32 uniform struct has a 16-byte WGSL layout footprint.
        resources.params = try self.createStorageBuffer(
            16,
            wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst,
        );
    }

    resources.staging = try self.createStorageBuffer(
        byte_size,
        wgpu.WGPUBufferUsage_MapRead | wgpu.WGPUBufferUsage_CopyDst,
    );

    var bind_entries: [4]wgpu.WGPUBindGroupEntry = undefined;
    for (&bind_entries) |*entry| entry.* = std.mem.zeroes(wgpu.WGPUBindGroupEntry);
    for (0..3) |binding| {
        bind_entries[binding].binding = @intCast(binding);
        bind_entries[binding].buffer = resources.storage[binding];
        bind_entries[binding].offset = 0;
        bind_entries[binding].size = wgpu.WGPU_WHOLE_SIZE;
    }
    const entry_count: usize = if (kernel == .add) 3 else 4;
    if (kernel == .saxpy) {
        bind_entries[3].binding = 3;
        bind_entries[3].buffer = resources.params;
        bind_entries[3].offset = 0;
        bind_entries[3].size = wgpu.WGPU_WHOLE_SIZE;
    }

    resources.bind_group = try self.createBindGroup(&wgpu.WGPUBindGroupDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .layout = self.pipelines[index].bind_group_layout,
        .entryCount = entry_count,
        .entries = bind_entries[0..].ptr,
    });
}

fn dispatchMany(
    self: *GpuContext,
    kernel: Kernel,
    n: usize,
    byte_size: usize,
    repetitions: usize,
) !void {
    const index = kernelIndex(kernel);
    const resources = &self.resources[index];
    const workgroup_count = workgroupCount(n);
    if (repetitions == 0 or !self.canRun(byte_size, workgroup_count)) {
        return error.GpuError;
    }
    const grid = try dispatchGrid(self, workgroup_count);

    self.beginErrorScope();
    const encoder = wgpu.wgpuDeviceCreateCommandEncoder(self.device, null) orelse {
        self.discardErrorScope();
        return error.GpuError;
    };

    for (0..repetitions) |_| {
        const pass = wgpu.wgpuCommandEncoderBeginComputePass(encoder, null) orelse {
            wgpu.wgpuCommandEncoderRelease(encoder);
            self.discardErrorScope();
            return error.GpuError;
        };
        wgpu.wgpuComputePassEncoderSetPipeline(pass, self.pipelines[index].pipeline);
        wgpu.wgpuComputePassEncoderSetBindGroup(pass, 0, resources.bind_group, 0, null);
        wgpu.wgpuComputePassEncoderDispatchWorkgroups(
            pass,
            grid.x,
            grid.y,
            1,
        );
        wgpu.wgpuComputePassEncoderEnd(pass);
        wgpu.wgpuComputePassEncoderRelease(pass);
    }

    wgpu.wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        resources.storage[2],
        0,
        resources.staging,
        0,
        @intCast(byte_size),
    );
    try self.submitRecorded(encoder);
    return;
}

fn dispatch(self: *GpuContext, kernel: Kernel, n: usize, byte_size: usize) !void {
    try dispatchMany(self, kernel, n, byte_size, 1);
}

fn readResult(self: *GpuContext, resources: *context_mod.BufferCache, out: []f32) !void {
    return self.readBuffer(resources.staging, std.mem.sliceAsBytes(out));
}

fn execute(
    self: *GpuContext,
    kernel: Kernel,
    out: []f32,
    a: []const f32,
    b: []const f32,
    alpha: ?f32,
) !void {
    if (out.len != a.len or a.len != b.len) return error.GpuError;
    if (out.len == 0) return;
    if (out.len > std.math.maxInt(usize) / @sizeOf(f32)) return error.GpuError;
    const byte_size = out.len * @sizeOf(f32);
    if (!self.canRun(byte_size, workgroupCount(out.len))) return error.GpuError;

    try ensurePipeline(self, kernel);
    try ensureResources(self, kernel, byte_size);

    const resources = &self.resources[kernelIndex(kernel)];
    self.writeBytes(resources.storage[0], std.mem.sliceAsBytes(a));
    self.writeBytes(resources.storage[1], std.mem.sliceAsBytes(b));
    if (kernel == .saxpy) {
        var alpha_value = alpha orelse return error.GpuError;
        self.writeBytes(resources.params, std.mem.asBytes(&alpha_value));
    }

    try dispatch(self, kernel, out.len, byte_size);
    try readResult(self, resources, out);
}

pub fn addWithContext(self: *GpuContext, out: []f32, a: []const f32, b: []const f32) !void {
    try execute(self, .add, out, a, b, null);
}

pub fn saxpyWithContext(self: *GpuContext, alpha: f32, out: []f32, x: []const f32, y: []const f32) !void {
    try execute(self, .saxpy, out, x, y, alpha);
}

/// Batch identical add dispatches for a steady-state kernel measurement.  It
/// still executes `iters` GPU dispatches, but intentionally uploads the two
/// invariant inputs and maps the result only once.  The ordinary add() path
/// above remains the end-to-end API with a readback per call.
pub fn addBatchedWithContext(
    self: *GpuContext,
    out: []f32,
    a: []const f32,
    b: []const f32,
    iters: usize,
) !void {
    if (out.len != a.len or a.len != b.len) return error.GpuError;
    if (out.len == 0 or iters == 0) return error.GpuError;
    if (out.len > std.math.maxInt(usize) / @sizeOf(f32)) return error.GpuError;
    const byte_size = out.len * @sizeOf(f32);
    if (!self.canRun(byte_size, workgroupCount(out.len))) return error.GpuError;

    try ensurePipeline(self, .add);
    try ensureResources(self, .add, byte_size);
    const resources = &self.resources[kernelIndex(.add)];
    self.writeBytes(resources.storage[0], std.mem.sliceAsBytes(a));
    self.writeBytes(resources.storage[1], std.mem.sliceAsBytes(b));
    try dispatchMany(self, .add, out.len, byte_size, iters);
    try readResult(self, resources, out);
}

/// Record a fallback from the comptime engine without exposing the context
/// cache to callers.
pub fn recordFallback(reason: []const u8) void {
    context_mod.recordFallback(reason);
}

pub fn clearFallbackReason() void {
    context_mod.clearFallbackReason();
}

pub fn lastFallbackReason() ?[]const u8 {
    return context_mod.lastFallbackReason();
}

pub fn add(out: []f32, a: []const f32, b: []const f32) !void {
    clearFallbackReason();
    const self = context_mod.global() catch |err| {
        if (lastFallbackReason() == null) recordFallback(@errorName(err));
        return err;
    };
    addWithContext(self, out, a, b) catch |err| {
        recordFallback(@errorName(err));
        return err;
    };
}

pub fn saxpy(alpha: f32, out: []f32, x: []const f32, y: []const f32) !void {
    clearFallbackReason();
    const self = context_mod.global() catch |err| {
        if (lastFallbackReason() == null) recordFallback(@errorName(err));
        return err;
    };
    saxpyWithContext(self, alpha, out, x, y) catch |err| {
        recordFallback(@errorName(err));
        return err;
    };
}

pub fn addBatched(out: []f32, a: []const f32, b: []const f32, iters: usize) !void {
    clearFallbackReason();
    const self = context_mod.global() catch |err| {
        if (lastFallbackReason() == null) recordFallback(@errorName(err));
        return err;
    };
    addBatchedWithContext(self, out, a, b, iters) catch |err| {
        recordFallback(@errorName(err));
        return err;
    };
}
