const std = @import("std");
const wgpu = @import("webgpu.zig");

/// All native WebGPU failures are surfaced through this error.  The caller of
/// the public engine API can then fall back to a CPU implementation.
pub const GpuError = error{GpuError};

/// Why the one-time capability probe failed.  The enum is intentionally more
/// useful to callers than a bare bool, while the accompanying `reason` string
/// remains stable and allocation-free for CLI diagnostics.
pub const ProbeFailure = enum {
    not_probed,
    none,
    instance_unavailable,
    adapter_unavailable,
    adapter_info_unavailable,
    device_unavailable,
    queue_unavailable,
    initialization_timeout,
    unsupported_type,

    pub fn reason(self: ProbeFailure) []const u8 {
        return switch (self) {
            .not_probed => "WebGPU capability has not been probed",
            .none => "WebGPU instance, adapter, device and queue are available",
            .instance_unavailable => "wgpuCreateInstance returned null (no WebGPU instance)",
            .adapter_unavailable => "WebGPU adapter request failed or returned no adapter",
            .adapter_info_unavailable => "wgpuAdapterGetInfo failed",
            .device_unavailable => "WebGPU device request failed or returned no device",
            .queue_unavailable => "wgpuDeviceGetQueue returned null",
            .initialization_timeout => "WebGPU initialization callback timed out",
            .unsupported_type => "gpu_webgpu is only available for f32 operations",
        };
    }
};

pub const ProbeResult = struct {
    available: bool,
    failure: ProbeFailure,
    reason: []const u8,
    adapter_backend_type: wgpu.WGPUBackendType = wgpu.WGPUBackendType_Undefined,
    adapter_vendor_id: u32 = 0,

    pub fn isAvailable(self: ProbeResult) bool {
        return self.available;
    }

    pub fn unavailable(failure: ProbeFailure) ProbeResult {
        return .{
            .available = false,
            .failure = failure,
            .reason = failure.reason(),
        };
    }
};

const InitFailure = error{
    InstanceUnavailable,
    AdapterUnavailable,
    AdapterInfoUnavailable,
    DeviceUnavailable,
    QueueUnavailable,
    InitializationTimeout,
};

pub const PipelineCache = struct {
    shader_module: wgpu.WGPUShaderModule = null,
    pipeline: wgpu.WGPUComputePipeline = null,
    pipeline_layout: wgpu.WGPUPipelineLayout = null,
    bind_group_layout: wgpu.WGPUBindGroupLayout = null,
};

pub const BufferCache = struct {
    byte_size: usize = 0,
    storage: [3]wgpu.WGPUBuffer = .{ null, null, null },
    params: wgpu.WGPUBuffer = null,
    staging: wgpu.WGPUBuffer = null,
    bind_group: wgpu.WGPUBindGroup = null,

    pub fn deinit(self: *BufferCache) void {
        if (self.bind_group) |handle| wgpu.wgpuBindGroupRelease(handle);
        self.bind_group = null;

        for (&self.storage) |*buffer| {
            if (buffer.*) |handle| wgpu.wgpuBufferRelease(handle);
            buffer.* = null;
        }
        if (self.params) |handle| wgpu.wgpuBufferRelease(handle);
        self.params = null;
        if (self.staging) |handle| wgpu.wgpuBufferRelease(handle);
        self.staging = null;
        self.byte_size = 0;
    }
};

const AdapterRequestState = struct {
    done: bool = false,
    status: wgpu.WGPURequestAdapterStatus = 0,
    adapter: wgpu.WGPUAdapter = null,
};

const DeviceRequestState = struct {
    done: bool = false,
    status: wgpu.WGPURequestDeviceStatus = 0,
    device: wgpu.WGPUDevice = null,
};

const ErrorScopeState = struct {
    done: bool = false,
    status: wgpu.WGPUPopErrorScopeStatus = 0,
    error_type: wgpu.WGPUErrorType = 0,
};

pub const MapState = struct {
    done: bool = false,
    status: wgpu.WGPUMapAsyncStatus = 0,
};

fn adapterCallback(
    status: wgpu.WGPURequestAdapterStatus,
    adapter: wgpu.WGPUAdapter,
    message: wgpu.WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const state = @as(*AdapterRequestState, @ptrCast(@alignCast(userdata1.?)));
    state.status = status;
    state.adapter = adapter;
    state.done = true;
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
    const state = @as(*DeviceRequestState, @ptrCast(@alignCast(userdata1.?)));
    state.status = status;
    state.device = device;
    state.done = true;
}

fn errorScopeCallback(
    status: wgpu.WGPUPopErrorScopeStatus,
    error_type: wgpu.WGPUErrorType,
    message: wgpu.WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = userdata2;
    const state = @as(*ErrorScopeState, @ptrCast(@alignCast(userdata1.?)));
    _ = message;
    state.status = status;
    state.error_type = error_type;
    state.done = true;
}

pub fn mapCallback(
    status: wgpu.WGPUMapAsyncStatus,
    message: wgpu.WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const state = @as(*MapState, @ptrCast(@alignCast(userdata1.?)));
    state.status = status;
    state.done = true;
}

pub const GpuContext = struct {
    allocator: std.mem.Allocator,
    instance: wgpu.WGPUInstance = null,
    adapter: wgpu.WGPUAdapter = null,
    device: wgpu.WGPUDevice = null,
    queue: wgpu.WGPUQueue = null,
    adapter_backend_type: wgpu.WGPUBackendType = wgpu.WGPUBackendType_Undefined,
    adapter_vendor_id: u32 = 0,

    // Index 0 = add, index 1 = saxpy.  Pipeline and bind-group resources are
    // kept alive across calls so a benchmark does not rebuild a pipeline per
    // dispatch.  Buffer resources are replaced only when the requested size
    // changes.
    pipelines: [2]PipelineCache = .{ .{}, .{} },
    resources: [2]BufferCache = .{ .{}, .{} },

    /// Explicit context construction keeps the original small `GpuError`
    /// contract.  The one-time probe below uses `initDetailed` so it can tell
    /// callers which initialization stage failed.
    pub fn init(allocator: std.mem.Allocator) !GpuContext {
        return initDetailed(allocator) catch return error.GpuError;
    }

    fn initDetailed(allocator: std.mem.Allocator) InitFailure!GpuContext {
        var self = GpuContext{ .allocator = allocator };
        errdefer self.deinit();

        self.instance = wgpu.wgpuCreateInstance(null) orelse
            return error.InstanceUnavailable;

        var adapter_state = AdapterRequestState{};
        _ = wgpu.wgpuInstanceRequestAdapter(self.instance, null, .{
            .nextInChain = null,
            .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
            .callback = adapterCallback,
            .userdata1 = @ptrCast(&adapter_state),
            .userdata2 = null,
        });
        self.waitForInitialization(&adapter_state) catch return error.InitializationTimeout;
        if (adapter_state.status != wgpu.WGPURequestAdapterStatus_Success or
            adapter_state.adapter == null)
        {
            return error.AdapterUnavailable;
        }
        self.adapter = adapter_state.adapter;

        var info = wgpu.WGPUAdapterInfo{
            .nextInChain = null,
            .vendor = emptyStringView(),
            .architecture = emptyStringView(),
            .device = emptyStringView(),
            .description = emptyStringView(),
            .backendType = wgpu.WGPUBackendType_Undefined,
            .adapterType = 0,
            .vendorID = 0,
            .deviceID = 0,
            .subgroupMinSize = 0,
            .subgroupMaxSize = 0,
        };
        const info_status = wgpu.wgpuAdapterGetInfo(self.adapter, &info);
        defer wgpu.wgpuAdapterInfoFreeMembers(info);
        if (info_status != wgpu.WGPUStatus_Success) return error.AdapterInfoUnavailable;
        self.adapter_backend_type = info.backendType;
        self.adapter_vendor_id = info.vendorID;

        var device_state = DeviceRequestState{};
        _ = wgpu.wgpuAdapterRequestDevice(self.adapter, null, .{
            .nextInChain = null,
            .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
            .callback = deviceCallback,
            .userdata1 = @ptrCast(&device_state),
            .userdata2 = null,
        });
        self.waitForInitialization(&device_state) catch return error.InitializationTimeout;
        if (device_state.status != wgpu.WGPURequestDeviceStatus_Success or
            device_state.device == null)
        {
            return error.DeviceUnavailable;
        }
        self.device = device_state.device;
        self.queue = wgpu.wgpuDeviceGetQueue(self.device) orelse
            return error.QueueUnavailable;

        return self;
    }

    /// Return the cached native WebGPU capability result.  The static method
    /// form keeps the probe discoverable as part of `GpuContext` while the
    /// cache itself lives at module scope.
    pub fn probe() ProbeResult {
        return probeCached();
    }

    pub fn deinit(self: *GpuContext) void {
        // Bind groups reference buffers and layouts, so release them first.
        for (&self.resources) |*resources| resources.deinit();
        for (&self.pipelines) |*pipeline| {
            if (pipeline.pipeline) |handle| wgpu.wgpuComputePipelineRelease(handle);
            pipeline.pipeline = null;
            if (pipeline.pipeline_layout) |handle| wgpu.wgpuPipelineLayoutRelease(handle);
            pipeline.pipeline_layout = null;
            if (pipeline.bind_group_layout) |handle| wgpu.wgpuBindGroupLayoutRelease(handle);
            pipeline.bind_group_layout = null;
            if (pipeline.shader_module) |handle| wgpu.wgpuShaderModuleRelease(handle);
            pipeline.shader_module = null;
        }

        if (self.queue) |handle| wgpu.wgpuQueueRelease(handle);
        self.queue = null;
        if (self.device) |handle| wgpu.wgpuDeviceRelease(handle);
        self.device = null;
        if (self.adapter) |handle| wgpu.wgpuAdapterRelease(handle);
        self.adapter = null;
        if (self.instance) |handle| wgpu.wgpuInstanceRelease(handle);
        self.instance = null;
    }

    /// Advance callbacks through ProcessEvents. This is the only async
    /// progress mechanism used by the native backend.
    pub fn pump(self: *GpuContext) void {
        if (self.instance) |instance| wgpu.wgpuInstanceProcessEvents(instance);
    }

    fn waitForInitialization(self: *GpuContext, state: anytype) error{InitializationTimeout}!void {
        var attempts: usize = 0;
        while (!state.done and attempts < 100_000) : (attempts += 1) self.pump();
        if (!state.done) return error.InitializationTimeout;
    }

    pub fn waitFor(self: *GpuContext, state: anytype) !void {
        var attempts: usize = 0;
        while (!state.done and attempts < 100_000) : (attempts += 1) self.pump();
        if (!state.done) return error.GpuError;
    }

    pub fn beginErrorScope(self: *GpuContext) void {
        wgpu.wgpuDevicePushErrorScope(self.device, wgpu.WGPUErrorFilter_Validation);
    }

    pub fn endErrorScope(self: *GpuContext) !void {
        var state = ErrorScopeState{};
        _ = wgpu.wgpuDevicePopErrorScope(self.device, .{
            .nextInChain = null,
            .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
            .callback = errorScopeCallback,
            .userdata1 = @ptrCast(&state),
            .userdata2 = null,
        });
        try self.waitFor(&state);
        if (state.status != wgpu.WGPUPopErrorScopeStatus_Success or
            state.error_type != wgpu.WGPUErrorType_NoError)
        {
            return error.GpuError;
        }
    }

    /// Used on an early-return path after beginErrorScope().  The error is
    /// deliberately ignored because the original operation is already failing.
    pub fn discardErrorScope(self: *GpuContext) void {
        self.endErrorScope() catch {};
    }

    pub fn createBuffer(
        self: *GpuContext,
        descriptor: *const wgpu.WGPUBufferDescriptor,
    ) !wgpu.WGPUBuffer {
        self.beginErrorScope();
        const result = wgpu.wgpuDeviceCreateBuffer(self.device, descriptor);
        self.endErrorScope() catch |err| {
            if (result) |handle| wgpu.wgpuBufferRelease(handle);
            return err;
        };
        return result orelse error.GpuError;
    }

    pub fn createBindGroupLayout(
        self: *GpuContext,
        descriptor: *const wgpu.WGPUBindGroupLayoutDescriptor,
    ) !wgpu.WGPUBindGroupLayout {
        self.beginErrorScope();
        const result = wgpu.wgpuDeviceCreateBindGroupLayout(self.device, descriptor);
        self.endErrorScope() catch |err| {
            if (result) |handle| wgpu.wgpuBindGroupLayoutRelease(handle);
            return err;
        };
        return result orelse error.GpuError;
    }

    pub fn createPipelineLayout(
        self: *GpuContext,
        descriptor: *const wgpu.WGPUPipelineLayoutDescriptor,
    ) !wgpu.WGPUPipelineLayout {
        self.beginErrorScope();
        const result = wgpu.wgpuDeviceCreatePipelineLayout(self.device, descriptor);
        self.endErrorScope() catch |err| {
            if (result) |handle| wgpu.wgpuPipelineLayoutRelease(handle);
            return err;
        };
        return result orelse error.GpuError;
    }

    pub fn createShaderModule(
        self: *GpuContext,
        descriptor: *const wgpu.WGPUShaderModuleDescriptor,
    ) !wgpu.WGPUShaderModule {
        self.beginErrorScope();
        const result = wgpu.wgpuDeviceCreateShaderModule(self.device, descriptor);
        self.endErrorScope() catch |err| {
            if (result) |handle| wgpu.wgpuShaderModuleRelease(handle);
            return err;
        };
        return result orelse error.GpuError;
    }

    pub fn createComputePipeline(
        self: *GpuContext,
        descriptor: *const wgpu.WGPUComputePipelineDescriptor,
    ) !wgpu.WGPUComputePipeline {
        self.beginErrorScope();
        const result = wgpu.wgpuDeviceCreateComputePipeline(self.device, descriptor);
        self.endErrorScope() catch |err| {
            if (result) |handle| wgpu.wgpuComputePipelineRelease(handle);
            return err;
        };
        return result orelse error.GpuError;
    }

    pub fn createBindGroup(
        self: *GpuContext,
        descriptor: *const wgpu.WGPUBindGroupDescriptor,
    ) !wgpu.WGPUBindGroup {
        self.beginErrorScope();
        const result = wgpu.wgpuDeviceCreateBindGroup(self.device, descriptor);
        self.endErrorScope() catch |err| {
            if (result) |handle| wgpu.wgpuBindGroupRelease(handle);
            return err;
        };
        return result orelse error.GpuError;
    }
};

fn emptyStringView() wgpu.WGPUStringView {
    return .{ .data = null, .length = wgpu.WGPU_STRLEN };
}

var global_context: ?GpuContext = null;
var probe_done: bool = false;
var cached_probe: ProbeResult = ProbeResult.unavailable(.not_probed);
var probe_override_for_testing: ?ProbeResult = null;
var probe_mutex: std.atomic.Mutex = .unlocked;

var fallback_reason: ?[]const u8 = null;
var fallback_mutex: std.atomic.Mutex = .unlocked;

fn lock(mutex: *std.atomic.Mutex) void {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
}

fn failureForInitError(err: InitFailure) ProbeFailure {
    return switch (err) {
        error.InstanceUnavailable => .instance_unavailable,
        error.AdapterUnavailable => .adapter_unavailable,
        error.AdapterInfoUnavailable => .adapter_info_unavailable,
        error.DeviceUnavailable => .device_unavailable,
        error.QueueUnavailable => .queue_unavailable,
        error.InitializationTimeout => .initialization_timeout,
    };
}

fn probeLocked() ProbeResult {
    if (probe_done) return cached_probe;
    if (probe_override_for_testing) |override| {
        cached_probe = override;
        probe_done = true;
        return cached_probe;
    }

    const context = GpuContext.initDetailed(std.heap.page_allocator) catch |err| {
        cached_probe = ProbeResult.unavailable(failureForInitError(err));
        probe_done = true;
        return cached_probe;
    };

    global_context = context;
    cached_probe = .{
        .available = true,
        .failure = .none,
        .reason = ProbeFailure.none.reason(),
        .adapter_backend_type = context.adapter_backend_type,
        .adapter_vendor_id = context.adapter_vendor_id,
    };
    probe_done = true;
    return cached_probe;
}

/// Probe native WebGPU exactly once per process.  Both success and failure
/// are cached, and the mutex also serializes initialization with `global()`.
/// A successful probe owns the context used by subsequent GPU operations, so
/// selection does not probe once and then silently initialize a different
/// context later.
fn probeCached() ProbeResult {
    lock(&probe_mutex);
    defer probe_mutex.unlock();
    return probeLocked();
}

/// Top-level alias for callers that do not retain the context type.
pub fn probe() ProbeResult {
    return probeCached();
}

/// The comptime-dispatched engine has a void API, so its GPU implementation
/// uses one process-local context.  Explicit callers/tests can still construct
/// and own a GpuContext directly.
pub fn global() !*GpuContext {
    lock(&probe_mutex);
    defer probe_mutex.unlock();

    const result = if (global_context != null) cached_probe else probeLocked();
    if (!result.available) {
        recordFallback(result.reason);
        return error.GpuError;
    }
    return &global_context.?;
}

/// Reset the process-local GPU state.  This is primarily useful for tests and
/// for applications that intentionally want to retry after changing their
/// driver environment; ordinary callers should rely on the one-time cache.
pub fn resetGlobal() void {
    lock(&probe_mutex);
    defer probe_mutex.unlock();

    if (global_context) |*context| context.deinit();
    global_context = null;
    probe_done = false;
    cached_probe = ProbeResult.unavailable(.not_probed);
    probe_override_for_testing = null;
    clearFallbackReason();
}

/// Test-only dependency injection for the unavailable-device path.  A
/// successful override is intentionally not supported because a fake result
/// must not manufacture a fake `GpuContext`; tests should inject failure only.
pub fn setProbeOverrideForTesting(override: ?ProbeResult) void {
    lock(&probe_mutex);
    defer probe_mutex.unlock();

    if (global_context) |*context| context.deinit();
    global_context = null;
    probe_done = false;
    cached_probe = ProbeResult.unavailable(.not_probed);
    probe_override_for_testing = override;
    clearFallbackReason();
}

/// Record why a void GPU engine call had to use its CPU implementation.  All
/// current callers pass static strings (`ProbeResult.reason` or `@errorName`),
/// so no allocation or lifetime management is needed.
pub fn recordFallback(reason_text: []const u8) void {
    lock(&fallback_mutex);
    defer fallback_mutex.unlock();
    fallback_reason = reason_text;
}

pub fn clearFallbackReason() void {
    lock(&fallback_mutex);
    defer fallback_mutex.unlock();
    fallback_reason = null;
}

pub fn lastFallbackReason() ?[]const u8 {
    lock(&fallback_mutex);
    defer fallback_mutex.unlock();
    return fallback_reason;
}
