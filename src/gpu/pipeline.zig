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

    var entries = layoutEntries(kernel);
    const bind_group_layout = try self.createBindGroupLayout(&wgpu.WGPUBindGroupLayoutDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .entryCount = if (kernel == .add) 3 else 4,
        .entries = entries[0..].ptr,
    });
    errdefer if (bind_group_layout) |handle| wgpu.wgpuBindGroupLayoutRelease(handle);

    var bind_group_layouts = [1]wgpu.WGPUBindGroupLayout{bind_group_layout};
    const pipeline_layout = try self.createPipelineLayout(&wgpu.WGPUPipelineLayoutDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = bind_group_layouts[0..].ptr,
        .immediateSize = 0,
    });
    errdefer if (pipeline_layout) |handle| wgpu.wgpuPipelineLayoutRelease(handle);

    const shader_code = if (kernel == .add) add_shader else saxpy_shader;
    var shader_source = wgpu.WGPUShaderSourceWGSL{
        .chain = .{
            .next = null,
            .sType = wgpu.WGPUSType_ShaderSourceWGSL,
        },
        .code = .{ .data = shader_code.ptr, .length = shader_code.len },
    };
    const shader_module = try self.createShaderModule(&wgpu.WGPUShaderModuleDescriptor{
        .nextInChain = @ptrCast(&shader_source.chain),
        .label = emptyStringView(),
    });
    errdefer if (shader_module) |handle| wgpu.wgpuShaderModuleRelease(handle);

    const pipeline = try self.createComputePipeline(&wgpu.WGPUComputePipelineDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .layout = pipeline_layout,
        .compute = .{
            .nextInChain = null,
            .module = shader_module,
            .entryPoint = stringView("main"),
            .constantCount = 0,
            .constants = null,
        },
    });

    cache.bind_group_layout = bind_group_layout;
    cache.pipeline_layout = pipeline_layout;
    cache.shader_module = shader_module;
    cache.pipeline = pipeline;
}

fn ensureResources(self: *GpuContext, kernel: Kernel, byte_size: usize) !void {
    const index = kernelIndex(kernel);
    const resources = &self.resources[index];
    if (resources.byte_size == byte_size and resources.bind_group != null) return;

    resources.deinit();
    resources.byte_size = byte_size;
    errdefer resources.deinit();

    var storage_descriptor = wgpu.WGPUBufferDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .usage = wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst,
        .size = @intCast(byte_size),
        .mappedAtCreation = 0,
    };
    for (0..3) |i| {
        if (i == 2) storage_descriptor.usage = wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc;
        resources.storage[i] = try self.createBuffer(&storage_descriptor);
    }

    if (kernel == .saxpy) {
        resources.params = try self.createBuffer(&wgpu.WGPUBufferDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .usage = wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst,
            // A one-f32 uniform struct has a 16-byte WGSL layout footprint.
            .size = 16,
            .mappedAtCreation = 0,
        });
    }

    resources.staging = try self.createBuffer(&wgpu.WGPUBufferDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .usage = wgpu.WGPUBufferUsage_MapRead | wgpu.WGPUBufferUsage_CopyDst,
        .size = @intCast(byte_size),
        .mappedAtCreation = 0,
    });

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

fn writeBytes(self: *GpuContext, buffer: wgpu.WGPUBuffer, bytes: []const u8) void {
    wgpu.wgpuQueueWriteBuffer(
        self.queue,
        buffer,
        0,
        @as(?*const anyopaque, @ptrCast(bytes.ptr)),
        bytes.len,
    );
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
    const command_buffer = wgpu.wgpuCommandEncoderFinish(encoder, null) orelse {
        wgpu.wgpuCommandEncoderRelease(encoder);
        self.discardErrorScope();
        return error.GpuError;
    };
    wgpu.wgpuCommandEncoderRelease(encoder);

    var commands = [1]wgpu.WGPUCommandBuffer{command_buffer};
    wgpu.wgpuQueueSubmit(self.queue, 1, commands[0..].ptr);
    self.endErrorScope() catch |err| {
        wgpu.wgpuCommandBufferRelease(command_buffer);
        return err;
    };
    wgpu.wgpuCommandBufferRelease(command_buffer);
    return;
}

fn dispatch(self: *GpuContext, kernel: Kernel, n: usize, byte_size: usize) !void {
    try dispatchMany(self, kernel, n, byte_size, 1);
}

fn readResult(self: *GpuContext, resources: *context_mod.BufferCache, out: []f32, byte_size: usize) !void {
    try mapRead(self, resources.staging, byte_size);
    const mapped = wgpu.wgpuBufferGetMappedRange(resources.staging, 0, byte_size) orelse {
        wgpu.wgpuBufferUnmap(resources.staging);
        return error.GpuError;
    };
    defer wgpu.wgpuBufferUnmap(resources.staging);
    const mapped_bytes = @as([*]const u8, @ptrCast(mapped))[0..byte_size];
    @memcpy(std.mem.sliceAsBytes(out), mapped_bytes);
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
    writeBytes(self, resources.storage[0], std.mem.sliceAsBytes(a));
    writeBytes(self, resources.storage[1], std.mem.sliceAsBytes(b));
    if (kernel == .saxpy) {
        var alpha_value = alpha orelse return error.GpuError;
        writeBytes(self, resources.params, std.mem.asBytes(&alpha_value));
    }

    try dispatch(self, kernel, out.len, byte_size);
    try readResult(self, resources, out, byte_size);
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
    writeBytes(self, resources.storage[0], std.mem.sliceAsBytes(a));
    writeBytes(self, resources.storage[1], std.mem.sliceAsBytes(b));
    try dispatchMany(self, .add, out.len, byte_size, iters);
    try readResult(self, resources, out, byte_size);
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

// Kept in this module so map callback setup cannot accidentally be changed to
// a wait-any-based implementation.
fn mapRead(self: *GpuContext, buffer: wgpu.WGPUBuffer, byte_size: usize) !void {
    var state = context_mod.MapState{};
    _ = wgpu.wgpuBufferMapAsync(buffer, wgpu.WGPUMapMode_Read, 0, byte_size, .{
        .nextInChain = null,
        .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
        .callback = context_mod.mapCallback,
        .userdata1 = @ptrCast(&state),
        .userdata2 = null,
    });
    try self.waitFor(&state);
    if (state.status != wgpu.WGPUMapAsyncStatus_Success) return error.GpuError;
}
