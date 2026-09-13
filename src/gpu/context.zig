const std = @import("std");
const wgpu = @import("webgpu.zig");

/// All native WebGPU failures are surfaced through this error.  The caller of
/// the public engine API can then fall back to a CPU implementation.
pub const GpuError = error{GpuError};

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

    pub fn init(allocator: std.mem.Allocator) !GpuContext {
        var self = GpuContext{ .allocator = allocator };
        errdefer self.deinit();

        self.instance = wgpu.wgpuCreateInstance(null) orelse return error.GpuError;

        var adapter_state = AdapterRequestState{};
        _ = wgpu.wgpuInstanceRequestAdapter(self.instance, null, .{
            .nextInChain = null,
            .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
            .callback = adapterCallback,
            .userdata1 = @ptrCast(&adapter_state),
            .userdata2 = null,
        });
        try self.waitFor(&adapter_state);
        if (adapter_state.status != wgpu.WGPURequestAdapterStatus_Success or
            adapter_state.adapter == null)
        {
            return error.GpuError;
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
        if (info_status != wgpu.WGPUStatus_Success) return error.GpuError;
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
        try self.waitFor(&device_state);
        if (device_state.status != wgpu.WGPURequestDeviceStatus_Success or
            device_state.device == null)
        {
            return error.GpuError;
        }
        self.device = device_state.device;
        self.queue = wgpu.wgpuDeviceGetQueue(self.device) orelse return error.GpuError;

        return self;
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

/// The comptime-dispatched engine has a void API, so its GPU implementation
/// uses one process-local context.  Explicit callers/tests can still construct
/// and own a GpuContext directly.
pub fn global() !*GpuContext {
    if (global_context == null) {
        global_context = try GpuContext.init(std.heap.page_allocator);
    }
    return &global_context.?;
}

pub fn resetGlobal() void {
    if (global_context) |*context| context.deinit();
    global_context = null;
}
