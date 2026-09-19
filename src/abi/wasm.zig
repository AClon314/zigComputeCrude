//! Browser entry point for the computeAccel WebGPU demo.
//!
//! The browser cannot block a JavaScript Promise with a native-style wait loop
//! when ASYNCIFY is disabled.  The GPU operation is therefore a small state
//! machine: ca_wasm_pump() calls wgpuInstanceProcessEvents() from the page's
//! animation-frame loop, and each AllowProcessEvents callback advances it.
//! The WebGPU declarations and the WGSL shader are still the T1 sources; this
//! file only supplies the browser entry/driver around that ABI.

const std = @import("std");
const wgpu = @import("computeAccel_gpu_webgpu");
const add_shader = @import("computeAccel_add_shader").source;

const element_count: usize = 1 << 20;
const status_capacity: usize = 512;

extern fn emscripten_get_now() f64;

var status_storage: [status_capacity]u8 = [_]u8{0} ** status_capacity;

fn setStatus(text: []const u8) void {
    const len = @min(text.len, status_storage.len - 1);
    @memcpy(status_storage[0..len], text[0..len]);
    status_storage[len] = 0;
}

fn setStatusFmt(comptime format: []const u8, args: anytype) void {
    const written = std.fmt.bufPrint(status_storage[0 .. status_storage.len - 1], format, args) catch {
        setStatus("GPU add: ERROR (status buffer too small)");
        return;
    };
    status_storage[written.len] = 0;
}

fn nowMs() f64 {
    return emscripten_get_now();
}

fn cpuSimdAdd(out: []f32, a: []const f32, b: []const f32) void {
    const V = @Vector(8, f32);
    var i: usize = 0;
    const chunks = a.len / 8;
    while (i < chunks * 8) : (i += 8) {
        const va: V = @as(*align(1) const V, @ptrCast(a.ptr + i)).*;
        const vb: V = @as(*align(1) const V, @ptrCast(b.ptr + i)).*;
        @as(*align(1) V, @ptrCast(out.ptr + i)).* = va + vb;
    }
    while (i < a.len) : (i += 1) out[i] = a[i] + b[i];
}

fn emptyStringView() wgpu.WGPUStringView {
    return .{ .data = null, .length = wgpu.WGPU_STRLEN };
}

fn stringView(comptime text: []const u8) wgpu.WGPUStringView {
    return .{ .data = text.ptr, .length = text.len };
}

const Runner = struct {
    // emscripten's malloc updates its JS heap views when memory grows.  The
    // freestanding Zig wasm allocator grows memory directly, which would
    // leave emdawnwebgpu's C-ABI view stale after these large allocations.
    allocator: std.mem.Allocator = std.heap.c_allocator,
    started: bool = false,
    done: bool = false,
    failed: bool = false,

    instance: wgpu.WGPUInstance = null,
    adapter: wgpu.WGPUAdapter = null,
    device: wgpu.WGPUDevice = null,
    queue: wgpu.WGPUQueue = null,

    bind_group_layout: wgpu.WGPUBindGroupLayout = null,
    pipeline_layout: wgpu.WGPUPipelineLayout = null,
    shader_module: wgpu.WGPUShaderModule = null,
    pipeline: wgpu.WGPUComputePipeline = null,
    bind_group: wgpu.WGPUBindGroup = null,
    input_a: wgpu.WGPUBuffer = null,
    input_b: wgpu.WGPUBuffer = null,
    output: wgpu.WGPUBuffer = null,
    staging: wgpu.WGPUBuffer = null,

    a: ?[]f32 = null,
    b: ?[]f32 = null,
    cpu_result: ?[]f32 = null,
    gpu_result: ?[]f32 = null,
    cpu_ms: f64 = 0,
    gpu_start_ms: f64 = 0,
    byte_size: usize = 0,

    fn fail(self: *Runner, stage: []const u8) void {
        if (self.done or self.failed) return;
        self.failed = true;
        setStatusFmt("GPU add: ERROR ({s})", .{stage});
    }

    fn setupAndDispatch(self: *Runner) !void {
        const byte_size = self.byte_size;

        var layout_entries: [3]wgpu.WGPUBindGroupLayoutEntry = undefined;
        for (&layout_entries) |*entry| {
            entry.* = std.mem.zeroes(wgpu.WGPUBindGroupLayoutEntry);
        }
        for (0..3) |binding| {
            layout_entries[binding].binding = @intCast(binding);
            layout_entries[binding].visibility = wgpu.WGPUShaderStage_Compute;
            layout_entries[binding].buffer.type = if (binding == 2)
                wgpu.WGPUBufferBindingType_Storage
            else
                wgpu.WGPUBufferBindingType_ReadOnlyStorage;
        }

        self.bind_group_layout = wgpu.wgpuDeviceCreateBindGroupLayout(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .entryCount = layout_entries.len,
            .entries = layout_entries[0..].ptr,
        }) orelse return error.GpuError;

        var bind_group_layouts = [1]wgpu.WGPUBindGroupLayout{self.bind_group_layout};
        self.pipeline_layout = wgpu.wgpuDeviceCreatePipelineLayout(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .bindGroupLayoutCount = bind_group_layouts.len,
            .bindGroupLayouts = bind_group_layouts[0..].ptr,
            .immediateSize = 0,
        }) orelse return error.GpuError;

        var shader_source = wgpu.WGPUShaderSourceWGSL{
            .chain = .{
                .next = null,
                .sType = wgpu.WGPUSType_ShaderSourceWGSL,
            },
            .code = .{ .data = add_shader.ptr, .length = add_shader.len },
        };
        self.shader_module = wgpu.wgpuDeviceCreateShaderModule(self.device, &.{
            .nextInChain = @ptrCast(&shader_source.chain),
            .label = emptyStringView(),
        }) orelse return error.GpuError;
        setStatus("loading: compute pipeline");
        self.pipeline = wgpu.wgpuDeviceCreateComputePipeline(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .layout = self.pipeline_layout,
            .compute = .{
                .nextInChain = null,
                .module = self.shader_module,
                .entryPoint = stringView("main"),
                .constantCount = 0,
                .constants = null,
            },
        }) orelse return error.GpuError;

        self.input_a = wgpu.wgpuDeviceCreateBuffer(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .usage = wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst,
            .size = @intCast(byte_size),
            .mappedAtCreation = 0,
        }) orelse return error.GpuError;
        self.input_b = wgpu.wgpuDeviceCreateBuffer(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .usage = wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst,
            .size = @intCast(byte_size),
            .mappedAtCreation = 0,
        }) orelse return error.GpuError;
        self.output = wgpu.wgpuDeviceCreateBuffer(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .usage = wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc,
            .size = @intCast(byte_size),
            .mappedAtCreation = 0,
        }) orelse return error.GpuError;
        self.staging = wgpu.wgpuDeviceCreateBuffer(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .usage = wgpu.WGPUBufferUsage_MapRead | wgpu.WGPUBufferUsage_CopyDst,
            .size = @intCast(byte_size),
            .mappedAtCreation = 0,
        }) orelse return error.GpuError;

        var bind_entries: [3]wgpu.WGPUBindGroupEntry = undefined;
        for (&bind_entries) |*entry| {
            entry.* = std.mem.zeroes(wgpu.WGPUBindGroupEntry);
        }
        const buffers = [3]wgpu.WGPUBuffer{ self.input_a, self.input_b, self.output };
        for (0..3) |binding| {
            bind_entries[binding].binding = @intCast(binding);
            bind_entries[binding].buffer = buffers[binding];
            bind_entries[binding].offset = 0;
            bind_entries[binding].size = wgpu.WGPU_WHOLE_SIZE;
        }
        self.bind_group = wgpu.wgpuDeviceCreateBindGroup(self.device, &.{
            .nextInChain = null,
            .label = emptyStringView(),
            .layout = self.bind_group_layout,
            .entryCount = bind_entries.len,
            .entries = bind_entries[0..].ptr,
        }) orelse return error.GpuError;

        const a = self.a.?;
        const b = self.b.?;
        wgpu.wgpuQueueWriteBuffer(
            self.queue,
            self.input_a,
            0,
            @as(?*const anyopaque, @ptrCast(a.ptr)),
            byte_size,
        );
        wgpu.wgpuQueueWriteBuffer(
            self.queue,
            self.input_b,
            0,
            @as(?*const anyopaque, @ptrCast(b.ptr)),
            byte_size,
        );

        const encoder = wgpu.wgpuDeviceCreateCommandEncoder(self.device, null) orelse
            return error.GpuError;
        const pass = wgpu.wgpuCommandEncoderBeginComputePass(encoder, null) orelse {
            wgpu.wgpuCommandEncoderRelease(encoder);
            return error.GpuError;
        };
        wgpu.wgpuComputePassEncoderSetPipeline(pass, self.pipeline);
        wgpu.wgpuComputePassEncoderSetBindGroup(pass, 0, self.bind_group, 0, null);
        // Keep the browser entry on the same 2D-grid path as native.  The
        // fixed demo size is below the guaranteed per-dimension limit, but
        // using the flattened form verifies the shared WGSL contract.
        const workgroups: usize = (element_count + 63) / 64;
        const dispatch_x: usize = @min(workgroups, 65_535);
        const dispatch_y: usize = (workgroups - 1) / dispatch_x + 1;
        wgpu.wgpuComputePassEncoderDispatchWorkgroups(
            pass,
            @intCast(dispatch_x),
            @intCast(dispatch_y),
            1,
        );
        wgpu.wgpuComputePassEncoderEnd(pass);
        wgpu.wgpuComputePassEncoderRelease(pass);
        wgpu.wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            self.output,
            0,
            self.staging,
            0,
            @intCast(byte_size),
        );
        const command_buffer = wgpu.wgpuCommandEncoderFinish(encoder, null) orelse {
            wgpu.wgpuCommandEncoderRelease(encoder);
            return error.GpuError;
        };
        wgpu.wgpuCommandEncoderRelease(encoder);

        var commands = [1]wgpu.WGPUCommandBuffer{command_buffer};
        self.gpu_start_ms = nowMs();
        wgpu.wgpuQueueSubmit(self.queue, 1, commands[0..].ptr);
        wgpu.wgpuCommandBufferRelease(command_buffer);

        _ = wgpu.wgpuBufferMapAsync(self.staging, wgpu.WGPUMapMode_Read, 0, byte_size, .{
            .nextInChain = null,
            .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
            .callback = mapCallback,
            .userdata1 = @ptrCast(self),
            .userdata2 = null,
        });
    }

    fn compare(self: *Runner) void {
        const cpu = self.cpu_result.?;
        const gpu = self.gpu_result.?;
        for (0..element_count) |i| {
            const cpu_bits: u32 = @bitCast(cpu[i]);
            const gpu_bits: u32 = @bitCast(gpu[i]);
            if (cpu_bits != gpu_bits) {
                setStatusFmt(
                    "GPU add: MISMATCH at i={} (gpu={d}, cpu={d})",
                    .{ i, gpu[i], cpu[i] },
                );
                self.failed = true;
                return;
            }
        }
        self.done = true;
        setStatusFmt(
            "GPU add: MATCH (n={}, gpu={d:.3} ms, cpu_simd={d:.3} ms)",
            .{ element_count, nowMs() - self.gpu_start_ms, self.cpu_ms },
        );
    }
};

var runner = Runner{};

fn adapterCallback(
    status: wgpu.WGPURequestAdapterStatus,
    adapter: wgpu.WGPUAdapter,
    message: wgpu.WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const self = @as(*Runner, @ptrCast(@alignCast(userdata1.?)));
    if (status != wgpu.WGPURequestAdapterStatus_Success or adapter == null) {
        self.fail("adapter unavailable");
        return;
    }
    self.adapter = adapter;
    setStatus("loading: WebGPU device");
    _ = wgpu.wgpuAdapterRequestDevice(self.adapter, null, .{
        .nextInChain = null,
        .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
        .callback = deviceCallback,
        .userdata1 = @ptrCast(self),
        .userdata2 = null,
    });
}

fn deviceCallback(
    status: wgpu.WGPURequestDeviceStatus,
    device: wgpu.WGPUDevice,
    message: wgpu.WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const self = @as(*Runner, @ptrCast(@alignCast(userdata1.?)));
    if (status != wgpu.WGPURequestDeviceStatus_Success or device == null) {
        self.fail("device unavailable");
        return;
    }
    self.device = device;
    self.queue = wgpu.wgpuDeviceGetQueue(device);
    if (self.queue == null) {
        self.fail("queue unavailable");
        return;
    }
    setStatus("loading: dispatching WGSL add");
    self.setupAndDispatch() catch self.fail("GPU dispatch");
}

fn mapCallback(
    status: wgpu.WGPUMapAsyncStatus,
    message: wgpu.WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const self = @as(*Runner, @ptrCast(@alignCast(userdata1.?)));
    if (status != wgpu.WGPUMapAsyncStatus_Success) {
        self.fail("readback map");
        return;
    }
    // A read-only GPUBuffer mapping must use the const WebGPU entry point in
    // emdawnwebgpu (native wgpu accepts the old spelling too).
    const mapped = wgpu.wgpuBufferGetConstMappedRange(self.staging, 0, self.byte_size) orelse {
        wgpu.wgpuBufferUnmap(self.staging);
        self.fail("readback range");
        return;
    };
    const mapped_bytes = @as([*]const u8, @ptrCast(mapped))[0..self.byte_size];
    @memcpy(std.mem.sliceAsBytes(self.gpu_result.?), mapped_bytes);
    wgpu.wgpuBufferUnmap(self.staging);
    self.compare();
}

/// Adapter preference for the browser path, set by the page before
/// `ca_wasm_main` runs (0 = auto/default, 1 = high-performance, 2 = low-power).
///
/// The page already chooses an adapter for `preinitializedWebGPUDevice`; passing
/// the same preference here keeps the C-ABI request from landing on a *different*
/// GPU on a hybrid laptop (browsers default to low-power, i.e. the integrated
/// one).  See docs/zig-gpu-spike.md §5.
var wasm_adapter_preference: wgpu.WGPUPowerPreference = wgpu.WGPUPowerPreference_Undefined;

export fn ca_wasm_set_adapter_preference(preference: u32) void {
    wasm_adapter_preference = switch (preference) {
        1 => wgpu.WGPUPowerPreference_HighPerformance,
        2 => wgpu.WGPUPowerPreference_LowPower,
        else => wgpu.WGPUPowerPreference_Undefined,
    };
}

/// Called by the C main() once the emcc module starts.
export fn ca_wasm_main() void {
    if (runner.started) return;
    runner.started = true;
    setStatus("loading: CPU add");

    const allocator = runner.allocator;
    runner.byte_size = element_count * @sizeOf(f32);
    runner.a = allocator.alloc(f32, element_count) catch {
        runner.fail("CPU allocation");
        return;
    };
    runner.b = allocator.alloc(f32, element_count) catch {
        runner.fail("CPU allocation");
        return;
    };
    runner.cpu_result = allocator.alloc(f32, element_count) catch {
        runner.fail("CPU allocation");
        return;
    };
    runner.gpu_result = allocator.alloc(f32, element_count) catch {
        runner.fail("CPU allocation");
        return;
    };

    for (0..element_count) |i| {
        runner.a.?[i] = @as(f32, @floatFromInt(i % 97)) * 0.25;
        runner.b.?[i] = @as(f32, @floatFromInt(i % 53)) * 0.5;
    }
    const cpu_start = nowMs();
    cpuSimdAdd(runner.cpu_result.?, runner.a.?, runner.b.?);
    runner.cpu_ms = nowMs() - cpu_start;

    runner.instance = wgpu.wgpuCreateInstance(null) orelse {
        runner.fail("instance unavailable");
        return;
    };
    setStatus("loading: WebGPU adapter");
    var adapter_options = wgpu.WGPURequestAdapterOptions.initial;
    adapter_options.powerPreference = wasm_adapter_preference;
    _ = wgpu.wgpuInstanceRequestAdapter(runner.instance, &adapter_options, .{
        .nextInChain = null,
        .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
        .callback = adapterCallback,
        .userdata1 = @ptrCast(&runner),
        .userdata2 = null,
    });
}

/// The page calls this once per animation frame.  It is deliberately the only
/// async progress mechanism: no WaitAny, device poll, or ASYNCIFY is involved.
export fn ca_wasm_pump() void {
    if (runner.instance) |instance| {
        if (!runner.done and !runner.failed) {
            wgpu.wgpuInstanceProcessEvents(instance);
        }
    }
}

/// NUL-terminated status consumed by shell.html through Module.HEAPU8.
export fn ca_wasm_status() [*:0]const u8 {
    return @ptrCast(&status_storage);
}
