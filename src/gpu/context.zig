const std = @import("std");
const wgpu = @import("webgpu.zig");

/// All native WebGPU failures are surfaced through this error.  The caller of
/// the public engine API can then fall back to a CPU implementation.
pub const GpuError = error{GpuError};

/// Wall-clock budget for one async callback (adapter/device/map/error-scope).
/// It only guards against a genuinely stuck device; it is not an estimate of
/// how long queued GPU work may take.
const wait_timeout_ns: i128 = 30 * 1_000_000_000;

fn nowNs() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// The subset of device limits needed by the compute backend.  A dispatch is
/// laid out as a 2D grid, so both axes must fit
/// `maxComputeWorkgroupsPerDimension`; each storage/staging allocation must fit
/// both buffer-size limits.
pub const GpuLimits = struct {
    maxComputeWorkgroupsPerDimension: u32 = 0,
    maxStorageBufferBindingSize: u64 = 0,
    maxBufferSize: u64 = 0,

    pub fn canRun(self: GpuLimits, n_bytes: usize, groups: usize) bool {
        if (n_bytes == 0 or groups == 0) return false;
        const byte_count: u64 = @intCast(n_bytes);
        if (byte_count > self.maxStorageBufferBindingSize or
            byte_count > self.maxBufferSize)
        {
            return false;
        }
        return self.workgroupGrid(groups) != null;
    }

    /// Flatten a linear workgroup count into the 2D grid used by every dispatch.
    /// WebGPU limits each axis independently, so the kernel reconstructs the
    /// linear index from `global_invocation_id` and `num_workgroups`.  Returns
    /// null when even the flattened grid cannot fit the device limit.
    pub fn workgroupGrid(self: GpuLimits, groups: usize) ?WorkgroupGrid {
        if (groups == 0) return null;
        const max_dimension: u64 = self.maxComputeWorkgroupsPerDimension;
        if (max_dimension == 0) return null;

        const group_count: u64 = @intCast(groups);
        const x = @min(group_count, max_dimension);
        // x is non-zero because groups and max_dimension were checked above.
        const y = (group_count - 1) / x + 1;
        if (y > max_dimension) return null;

        return .{
            .x = @intCast(x),
            .y = @intCast(y),
        };
    }
};

pub const WorkgroupGrid = struct {
    x: u32,
    y: u32,
};

/// Why the one-time capability probe failed.  The enum is intentionally more
/// useful to callers than a bare bool, while the accompanying `reason` string
/// remains stable and allocation-free for CLI diagnostics.
pub const ProbeFailure = enum {
    not_probed,
    none,
    instance_unavailable,
    adapter_unavailable,
    adapter_info_unavailable,
    adapter_limits_unavailable,
    device_unavailable,
    device_limits_unavailable,
    queue_unavailable,
    initialization_timeout,
    unsupported_type,
    request_exceeds_limits,

    pub fn reason(self: ProbeFailure) []const u8 {
        return switch (self) {
            .not_probed => "WebGPU capability has not been probed",
            .none => "WebGPU instance, adapter, device and queue are available",
            .instance_unavailable => "wgpuCreateInstance returned null (no WebGPU instance)",
            .adapter_unavailable => "WebGPU adapter request failed or returned no adapter",
            .adapter_info_unavailable => "wgpuAdapterGetInfo failed",
            .adapter_limits_unavailable => "wgpuAdapterGetLimits failed",
            .device_unavailable => "WebGPU device request failed or returned no device",
            .device_limits_unavailable => "wgpuDeviceGetLimits failed",
            .queue_unavailable => "wgpuDeviceGetQueue returned null",
            .initialization_timeout => "WebGPU initialization callback timed out",
            .unsupported_type => "gpu_webgpu is only available for f32 operations",
            .request_exceeds_limits => "requested GPU size exceeds WebGPU limits",
        };
    }
};

pub const ProbeResult = struct {
    available: bool,
    failure: ProbeFailure,
    reason: []const u8,
    limits: GpuLimits = .{},
    adapter_backend_type: wgpu.WGPUBackendType = wgpu.WGPUBackendType_Undefined,
    adapter_vendor_id: u32 = 0,
    /// Which preference produced this adapter (`.auto` is the C default).
    preference: AdapterPreference = .auto,
    /// 完整选择（含 force_fallback），用于日志/消融表。
    selection: AdapterSelection = .{},
    /// Fixed-size copy of `WGPUAdapterInfo`, so the result stays valid after
    /// the raw string views are freed and needs no allocation.
    adapter_info: AdapterInfo = .{},

    pub fn isAvailable(self: ProbeResult) bool {
        return self.available;
    }

    pub fn canRun(self: ProbeResult, n_bytes: usize, groups: usize) bool {
        return self.available and self.limits.canRun(n_bytes, groups);
    }

    pub fn unavailable(failure: ProbeFailure) ProbeResult {
        return .{
            .available = false,
            .failure = failure,
            .reason = failure.reason(),
        };
    }
};

/// How the process-local context picks a WebGPU adapter.
///
/// This is an **init-time** switch: it is read when a context/probe is first
/// created, and each preference is cached separately, so probing or opening two
/// preferences in one run (e.g. an adapter ablation in the CLI) yields two
/// independent contexts instead of invalidating the first one.
///
/// Measured consequence on a hybrid laptop (Ryzen 5 5600H + Vega iGPU +
/// RTX 3050, wgpu-native v29.0.1.1): the C default (`Undefined`/`.auto`) resolves
/// to the *integrated* adapter, `.high_performance` to the discrete one; GEMM and
/// the chained pipeline are 1.4–2.5x faster on the discrete adapter, while a
/// transfer-bound reduce is unchanged.  See `docs/zig-gpu-spike.md` §5.
pub const AdapterPreference = enum {
    /// Leave WebGPU's default in place (what the C API does with
    /// `powerPreference = Undefined`).
    auto,
    high_performance,
    low_power,

    pub fn toWgpu(self: AdapterPreference) wgpu.WGPUPowerPreference {
        return switch (self) {
            .auto => wgpu.WGPUPowerPreference_Undefined,
            .high_performance => wgpu.WGPUPowerPreference_HighPerformance,
            .low_power => wgpu.WGPUPowerPreference_LowPower,
        };
    }

    pub fn name(self: AdapterPreference) []const u8 {
        return switch (self) {
            .auto => "auto",
            .high_performance => "high-performance",
            .low_power => "low-power",
        };
    }

    fn slot(self: AdapterPreference) usize {
        return switch (self) {
            .auto => 0,
            .high_performance => 1,
            .low_power => 2,
        };
    }
};

/// `AdapterSelection` 的缓存槽：3 个偏好 + 1 个软件 fallback 槽。
fn selectionSlot(selection: AdapterSelection) usize {
    if (selection.force_fallback) return 3;
    return selection.preference.slot();
}

/// One adapter selection.  A struct (rather than a bare enum) so extra knobs can
/// be added without changing call sites.
pub const AdapterSelection = struct {
    preference: AdapterPreference = .auto,
    /// Ask for the implementation's fallback (software) adapter — wgpu-native 会
    /// 给 Mesa llvmpipe，浏览器端 `forceFallbackAdapter` 会给 SwiftShader。
    /// 用途：验证"软件适配器上也能跑对"（慢，但 ABI/对拍契约一致）。
    force_fallback: bool = false,

    pub fn label(self: AdapterSelection) []const u8 {
        if (self.force_fallback) return "fallback(software)";
        return self.preference.name();
    }
};

/// Fixed-size, allocation-free copy of the selected adapter's `WGPUAdapterInfo`.
pub const AdapterInfo = struct {
    description_buf: [96]u8 = [_]u8{0} ** 96,
    description_len: u8 = 0,
    /// 浏览器（emdawnwebgpu）只填 vendor/architecture，`description`/`device`
    /// 因隐私是空的，所以分类必须能看这两个字段。native 侧则反过来更全。
    vendor_buf: [32]u8 = [_]u8{0} ** 32,
    vendor_len: u8 = 0,
    architecture_buf: [32]u8 = [_]u8{0} ** 32,
    architecture_len: u8 = 0,
    vendor_id: u32 = 0,
    device_id: u32 = 0,
    adapter_type: wgpu.WGPUAdapterType = wgpu.WGPUAdapterType_Unknown,
    backend_type: wgpu.WGPUBackendType = wgpu.WGPUBackendType_Undefined,

    /// The adapter's description (e.g. "NVIDIA GeForce RTX 3050 Laptop GPU").
    /// Takes a pointer receiver on purpose: the returned slice points into
    /// `self`, so a by-value receiver would hand back a dangling slice.
    pub fn description(self: *const AdapterInfo) []const u8 {
        return self.description_buf[0..self.description_len];
    }

    pub fn vendor(self: *const AdapterInfo) []const u8 {
        return self.vendor_buf[0..self.vendor_len];
    }

    pub fn architecture(self: *const AdapterInfo) []const u8 {
        return self.architecture_buf[0..self.architecture_len];
    }

    /// 硬件适配器的粗分类（来自驱动的 `adapterType`，native 侧可信；
    /// 浏览器侧 emdawnwebgpu 一律报 Unknown，见 `isSoftware`）。
    pub fn kind(self: AdapterInfo) AdapterKind {
        return switch (self.adapter_type) {
            wgpu.WGPUAdapterType_DiscreteGPU => .discrete,
            wgpu.WGPUAdapterType_IntegratedGPU => .integrated,
            wgpu.WGPUAdapterType_CPU => .software,
            else => .unknown,
        };
    }

    /// 是否是软件光栅化/模拟器（CPU 上跑的"GPU"）。
    ///
    /// 判据分两级：
    ///  1. **权威**：驱动报 `adapterType == CPU`（native 的 llvmpipe 与浏览器的
    ///     SwiftShader 都会报，实测 Chrome 里只有 fallback 适配器填了 type=3）；
    ///  2. **启发式**：`vendor/architecture/description` 命中已知软件实现名。
    ///     因为 emdawnwebgpu 不填 `adapterType`（实测 nvidia/amd/swiftshader 里
    ///     只有 swiftshader 有 type），浏览器里没有别的办法分辨"软件适配器"。
    ///     名字表只收确定是软件实现的关键词，宁漏不误。
    pub fn isSoftware(self: *const AdapterInfo) bool {
        if (self.kind() == .software) return true;
        const software_markers = [_][]const u8{
            "swiftshader",            // Chrome/Dawn 的软件 Vulkan
            "llvmpipe",               // Mesa 软件 Vulkan
            "lavapipe",               // Mesa 软件 Vulkan（LunarG 名）
            "softpipe",               // Mesa 软件 GL
            "software rasterizer",    // 通用描述
            "software adapter",       // 通用描述
            "microsoft basic render", // D3D WARP/基本渲染驱动
        };
        for (software_markers) |marker| {
            if (containsIgnoreCase(self.vendor(), marker)) return true;
            if (containsIgnoreCase(self.architecture(), marker)) return true;
            if (containsIgnoreCase(self.description(), marker)) return true;
        }
        return false;
    }

    /// 给日志/CLI 用的一行式路径标签：`gpu/discrete`、`gpu/integrated`、
    /// `gpu/software` 或 `gpu/unknown`。
    pub fn pathLabel(self: *const AdapterInfo) []const u8 {
        if (self.isSoftware()) return "gpu/software";
        return switch (self.kind()) {
            .discrete => "gpu/discrete",
            .integrated => "gpu/integrated",
            .software => "gpu/software",
            .unknown => "gpu/unknown",
        };
    }

    pub fn adapterTypeName(self: AdapterInfo) []const u8 {
        return switch (self.adapter_type) {
            wgpu.WGPUAdapterType_DiscreteGPU => "discrete",
            wgpu.WGPUAdapterType_IntegratedGPU => "integrated",
            wgpu.WGPUAdapterType_CPU => "cpu",
            else => "unknown",
        };
    }

    pub fn isDiscrete(self: AdapterInfo) bool {
        return self.adapter_type == wgpu.WGPUAdapterType_DiscreteGPU;
    }

    fn fromRaw(raw: wgpu.WGPUAdapterInfo) AdapterInfo {
        var info = AdapterInfo{
            .vendor_id = raw.vendorID,
            .device_id = raw.deviceID,
            .adapter_type = @intCast(raw.adapterType),
            .backend_type = raw.backendType,
        };
        copyString(raw.description, &info.description_buf, &info.description_len);
        copyString(raw.vendor, &info.vendor_buf, &info.vendor_len);
        copyString(raw.architecture, &info.architecture_buf, &info.architecture_len);
        return info;
    }
};

/// Human-readable backend name for diagnostics (the native Linux path is Vulkan
/// today; the others exist so a browser/other-platform report stays readable).
pub fn backendTypeName(backend_type: wgpu.WGPUBackendType) []const u8 {
    return switch (backend_type) {
        wgpu.WGPUBackendType_Null => "null",
        wgpu.WGPUBackendType_WebGPU => "webgpu",
        wgpu.WGPUBackendType_D3D11 => "d3d11",
        wgpu.WGPUBackendType_D3D12 => "d3d12",
        wgpu.WGPUBackendType_Metal => "metal",
        wgpu.WGPUBackendType_Vulkan => "vulkan",
        wgpu.WGPUBackendType_OpenGL => "opengl",
        wgpu.WGPUBackendType_OpenGLES => "opengles",
        else => "undefined",
    };
}

pub const AdapterKind = enum { discrete, integrated, software, unknown };

fn copyString(view: wgpu.WGPUStringView, out: []u8, out_len: *u8) void {
    const text = stringViewSlice(view);
    const len = @min(text.len, out.len);
    @memcpy(out[0..len], text[0..len]);
    out_len.* = @intCast(len);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn stringViewSlice(view: wgpu.WGPUStringView) []const u8 {
    const data = view.data orelse return "";
    if (view.length == wgpu.WGPU_STRLEN) return std.mem.sliceTo(data, 0);
    return data[0..view.length];
}

const InitFailure = error{
    InstanceUnavailable,
    AdapterUnavailable,
    AdapterInfoUnavailable,
    AdapterLimitsUnavailable,
    DeviceUnavailable,
    DeviceLimitsUnavailable,
    QueueUnavailable,
    InitializationTimeout,
};

pub const PipelineCache = struct {
    shader_module: wgpu.WGPUShaderModule = null,
    pipeline: wgpu.WGPUComputePipeline = null,
    pipeline_layout: wgpu.WGPUPipelineLayout = null,
    /// Owned when set.  Kernel variants that share one bind group layout (GEMM
    /// simple/tiled, reduce sum/max) leave this null and keep the shared layout
    /// in their own cache, so `deinit` never double-releases it.
    bind_group_layout: wgpu.WGPUBindGroupLayout = null,

    pub fn deinit(self: *PipelineCache) void {
        if (self.pipeline) |handle| wgpu.wgpuComputePipelineRelease(handle);
        self.pipeline = null;
        if (self.pipeline_layout) |handle| wgpu.wgpuPipelineLayoutRelease(handle);
        self.pipeline_layout = null;
        if (self.bind_group_layout) |handle| wgpu.wgpuBindGroupLayoutRelease(handle);
        self.bind_group_layout = null;
        if (self.shader_module) |handle| wgpu.wgpuShaderModuleRelease(handle);
        self.shader_module = null;
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
    limits: GpuLimits = .{},
    adapter_backend_type: wgpu.WGPUBackendType = wgpu.WGPUBackendType_Undefined,
    adapter_vendor_id: u32 = 0,
    adapter_info: AdapterInfo = .{},

    /// Explicit context construction keeps the original small `GpuError`
    /// contract.  The one-time probe below uses `initDetailed` so it can tell
    /// callers which initialization stage failed.
    pub fn init(allocator: std.mem.Allocator) !GpuContext {
        return initDetailed(allocator, .{}) catch return error.GpuError;
    }

    /// Same as `init`, but with an explicit adapter selection (init-time switch).
    pub fn initWithAdapter(
        allocator: std.mem.Allocator,
        selection: AdapterSelection,
    ) !GpuContext {
        return initDetailed(allocator, selection) catch return error.GpuError;
    }

    fn initDetailed(
        allocator: std.mem.Allocator,
        selection: AdapterSelection,
    ) InitFailure!GpuContext {
        var self = GpuContext{ .allocator = allocator };
        errdefer self.deinit();

        self.instance = wgpu.wgpuCreateInstance(null) orelse
            return error.InstanceUnavailable;

        var adapter_state = AdapterRequestState{};
        var adapter_options = wgpu.WGPURequestAdapterOptions.initial;
        adapter_options.powerPreference = selection.preference.toWgpu();
        adapter_options.forceFallbackAdapter = if (selection.force_fallback) 1 else 0;
        _ = wgpu.wgpuInstanceRequestAdapter(self.instance, &adapter_options, .{
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
        self.adapter_info = AdapterInfo.fromRaw(info);

        // Query the adapter before requesting a device and query the device
        // again below.  The device limits are authoritative for the context;
        // keeping both calls here also catches an ABI/driver failure before a
        // size-dependent dispatch reaches validation.
        var adapter_limits = std.mem.zeroes(wgpu.WGPULimits);
        if (wgpu.wgpuAdapterGetLimits(self.adapter, &adapter_limits) !=
            wgpu.WGPUStatus_Success)
        {
            return error.AdapterLimitsUnavailable;
        }

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

        var device_limits = std.mem.zeroes(wgpu.WGPULimits);
        if (wgpu.wgpuDeviceGetLimits(self.device, &device_limits) !=
            wgpu.WGPUStatus_Success)
        {
            return error.DeviceLimitsUnavailable;
        }
        self.limits = .{
            .maxComputeWorkgroupsPerDimension = device_limits.maxComputeWorkgroupsPerDimension,
            .maxStorageBufferBindingSize = device_limits.maxStorageBufferBindingSize,
            .maxBufferSize = device_limits.maxBufferSize,
        };

        self.queue = wgpu.wgpuDeviceGetQueue(self.device) orelse
            return error.QueueUnavailable;

        return self;
    }

    /// Return the cached native WebGPU capability result for the process
    /// default adapter selection.  The static method form keeps the probe
    /// discoverable as part of `GpuContext` while the cache itself lives at
    /// module scope.
    pub fn probe() ProbeResult {
        return probeCachedWith(adapterSelection());
    }

    /// Return whether this context can represent both the requested buffers
    /// and the flattened 2D dispatch grid without submitting a validation
    /// error.
    pub fn canRun(self: *const GpuContext, n_bytes: usize, groups: usize) bool {
        return self.limits.canRun(n_bytes, groups);
    }

    /// Which physical adapter this context actually opened.  WebGPU's default
    /// power preference is *not* "the fastest GPU": on a hybrid machine it
    /// resolves to the integrated adapter, so callers that care should log this
    /// (or pass `AdapterSelection{ .preference = .high_performance }`).
    pub fn adapterInfo(self: *const GpuContext) AdapterInfo {
        return self.adapter_info;
    }

    pub fn deinit(self: *GpuContext) void {
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

    /// Wait for an async callback with a wall-clock budget.
    ///
    /// A fixed iteration count is the wrong unit here: the number of
    /// ProcessEvents calls needed is proportional to the GPU work queued
    /// before the callback (a 5-dispatch GEMM batch on the iGPU needs far more
    /// than an add), so a constant cap silently expires while the device is
    /// still busy.  The budget below is a real timeout, not a work estimate.
    fn waitForInitialization(self: *GpuContext, state: anytype) error{InitializationTimeout}!void {
        const start = nowNs();
        while (!state.done) {
            self.pump();
            if (nowNs() - start > wait_timeout_ns) return error.InitializationTimeout;
        }
    }

    pub fn waitFor(self: *GpuContext, state: anytype) !void {
        const start = nowNs();
        while (!state.done) {
            self.pump();
            if (nowNs() - start > wait_timeout_ns) return error.GpuError;
        }
    }

    /// Map a MapRead buffer with ProcessEvents only, copy its bytes into
    /// `out`, and unmap before returning.  No WaitAny/device-poll path is
    /// involved; gemm/reduce reuse this so readback cannot drift per kernel.
    ///
    /// If the wait times out, the pending mapping is aborted with
    /// `wgpuBufferUnmap` so the buffer returns to `Idle`; otherwise a later
    /// `wgpuQueueSubmit` would fail validation with "buffer is still mapped"
    /// (a fatal error in wgpu-native, not a catchable one).
    pub fn readBuffer(self: *GpuContext, buffer: wgpu.WGPUBuffer, out: []u8) !void {
        if (out.len == 0) return;
        var state = MapState{};
        _ = wgpu.wgpuBufferMapAsync(buffer, wgpu.WGPUMapMode_Read, 0, out.len, .{
            .nextInChain = null,
            .mode = wgpu.WGPUCallbackMode_AllowProcessEvents,
            .callback = mapCallback,
            .userdata1 = @ptrCast(&state),
            .userdata2 = null,
        });
        self.waitFor(&state) catch |err| {
            // Cancel a mapping stuck in `Waiting` so the buffer can be reused.
            wgpu.wgpuBufferUnmap(buffer);
            return err;
        };
        if (state.status != wgpu.WGPUMapAsyncStatus_Success) return error.GpuError;

        const mapped = wgpu.wgpuBufferGetMappedRange(buffer, 0, out.len) orelse {
            wgpu.wgpuBufferUnmap(buffer);
            return error.GpuError;
        };
        defer wgpu.wgpuBufferUnmap(buffer);
        const mapped_bytes = @as([*]const u8, @ptrCast(mapped))[0..out.len];
        @memcpy(out, mapped_bytes);
    }

    /// Create an owned storage/uniform buffer with the context's error scope.
    /// Every kernel module uses this instead of repeating the descriptor.
    pub fn createStorageBuffer(
        self: *GpuContext,
        byte_size: usize,
        usage: wgpu.WGPUBufferUsage,
    ) !wgpu.WGPUBuffer {
        return self.createBuffer(&wgpu.WGPUBufferDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .usage = usage,
            .size = @intCast(byte_size),
            .mappedAtCreation = 0,
        });
    }

    /// Copy host bytes into a buffer, used for every storage/uniform upload.
    pub fn writeBytes(self: *GpuContext, buffer: wgpu.WGPUBuffer, bytes: []const u8) void {
        wgpu.wgpuQueueWriteBuffer(
            self.queue,
            buffer,
            0,
            @as(?*const anyopaque, @ptrCast(bytes.ptr)),
            bytes.len,
        );
    }

    /// Compile one WGSL module into `slot` with the given (possibly shared)
    /// bind group layout.  `entry_point` is comptime so the string view stays
    /// allocation-free, matching the rest of the ABI layer.
    pub fn createKernelPipeline(
        self: *GpuContext,
        slot: *PipelineCache,
        shader_code: []const u8,
        comptime entry_point: []const u8,
        bind_group_layout: wgpu.WGPUBindGroupLayout,
    ) !void {
        errdefer slot.deinit();

        var bind_group_layouts = [1]wgpu.WGPUBindGroupLayout{bind_group_layout};
        slot.pipeline_layout = try self.createPipelineLayout(&wgpu.WGPUPipelineLayoutDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .bindGroupLayoutCount = bind_group_layouts.len,
            .bindGroupLayouts = bind_group_layouts[0..].ptr,
            .immediateSize = 0,
        });

        var shader_source = wgpu.WGPUShaderSourceWGSL{
            .chain = .{
                .next = null,
                .sType = wgpu.WGPUSType_ShaderSourceWGSL,
            },
            .code = .{ .data = shader_code.ptr, .length = shader_code.len },
        };
        slot.shader_module = try self.createShaderModule(&wgpu.WGPUShaderModuleDescriptor{
            .nextInChain = @ptrCast(&shader_source.chain),
            .label = emptyStringView(),
        });

        slot.pipeline = try self.createComputePipeline(&wgpu.WGPUComputePipelineDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .layout = slot.pipeline_layout,
            .compute = .{
                .nextInChain = null,
                .module = slot.shader_module,
                .entryPoint = stringView(entry_point),
                .constantCount = 0,
                .constants = null,
            },
        });
    }

    /// Finish and submit an encoder, then check the error scope that the caller
    /// pushed before recording.  This is the single submit path for all kernels.
    pub fn submitRecorded(self: *GpuContext, encoder: wgpu.WGPUCommandEncoder) !void {
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
    }

    /// One-off device-to-device copy followed by a submit; used by blocking
    /// readbacks that are not part of a `Chain`.
    pub fn copyBufferToStaging(
        self: *GpuContext,
        src: wgpu.WGPUBuffer,
        dst: wgpu.WGPUBuffer,
        byte_size: usize,
    ) !void {
        self.beginErrorScope();
        const encoder = wgpu.wgpuDeviceCreateCommandEncoder(self.device, null) orelse {
            self.discardErrorScope();
            return error.GpuError;
        };
        wgpu.wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, dst, 0, @intCast(byte_size));
        try self.submitRecorded(encoder);
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

/// Public so kernel/buffer modules can build descriptors without duplicating
/// the empty-label convention.
pub fn emptyStringView() wgpu.WGPUStringView {
    return .{ .data = null, .length = wgpu.WGPU_STRLEN };
}

fn stringView(comptime text: []const u8) wgpu.WGPUStringView {
    return .{ .data = text.ptr, .length = text.len };
}

var probe_override_for_testing: ?ProbeResult = null;
var probe_mutex: std.atomic.Mutex = .unlocked;

/// Default selection used by `probe()` / `global()`.  Set it before the first
/// GPU use (`setAdapterSelection`); a library consumer normally leaves it at
/// `.auto`.
var default_selection: AdapterSelection = .{};

/// One cached probe per `AdapterPreference` (3 slots).  Keeping them separate
/// means an adapter ablation can open the iGPU and the dGPU in the same process
/// without pulling the device out from under the first context.
const adapter_slot_count = 4;
const ProbeSlot = struct {
    done: bool = false,
    result: ProbeResult = ProbeResult.unavailable(.not_probed),
    context: ?GpuContext = null,
};
var probe_slots: [adapter_slot_count]ProbeSlot = .{ .{}, .{}, .{}, .{} };

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
        error.AdapterLimitsUnavailable => .adapter_limits_unavailable,
        error.DeviceUnavailable => .device_unavailable,
        error.DeviceLimitsUnavailable => .device_limits_unavailable,
        error.QueueUnavailable => .queue_unavailable,
        error.InitializationTimeout => .initialization_timeout,
    };
}

fn probeLocked() ProbeResult {
    return probeWithLocked(default_selection);
}

/// Probe (and keep) the context for one preference.  Called with `probe_mutex`
/// held.  A successful probe stores the context in that preference's slot, which
/// is what the void-API engine (`global()`) reads back.
fn probeWithLocked(selection: AdapterSelection) ProbeResult {
    const slot = &probe_slots[selectionSlot(selection)];
    if (slot.done) return slot.result;
    if (probe_override_for_testing) |override| {
        slot.result = override;
        slot.done = true;
        return slot.result;
    }

    const context = GpuContext.initDetailed(std.heap.page_allocator, selection) catch |err| {
        slot.result = ProbeResult.unavailable(failureForInitError(err));
        slot.done = true;
        return slot.result;
    };

    slot.context = context;
    slot.result = .{
        .available = true,
        .failure = .none,
        .reason = ProbeFailure.none.reason(),
        .limits = context.limits,
        .adapter_backend_type = context.adapter_backend_type,
        .adapter_vendor_id = context.adapter_vendor_id,
        .preference = selection.preference,
        .selection = selection,
        .adapter_info = context.adapter_info,
    };
    slot.done = true;
    return slot.result;
}

/// Probe native WebGPU exactly once per adapter preference.  Both success and
/// failure are cached, and the mutex also serializes initialization with
/// `global()`.  A successful probe owns the context used by subsequent GPU
/// operations, so selection does not probe once and then silently initialize a
/// different context later.
fn probeCachedWith(selection: AdapterSelection) ProbeResult {
    lock(&probe_mutex);
    defer probe_mutex.unlock();
    return probeWithLocked(selection);
}

/// Set the adapter preference used by `probe()` / `global()` / the
/// comptime-dispatched engine.
///
/// This is an init-time switch: call it before the first GPU use.  Contexts
/// already created for an earlier preference stay valid (each preference is
/// cached separately), but they are not reused by `global()` afterwards.
/// Returns the previous selection.
pub fn setAdapterSelection(selection: AdapterSelection) AdapterSelection {
    lock(&probe_mutex);
    defer probe_mutex.unlock();
    const previous = default_selection;
    default_selection = selection;
    return previous;
}

/// The selection `probe()` / `global()` currently use.
pub fn adapterSelection() AdapterSelection {
    lock(&probe_mutex);
    defer probe_mutex.unlock();
    return default_selection;
}

/// Top-level alias for callers that do not retain the context type.
pub fn probe() ProbeResult {
    lock(&probe_mutex);
    defer probe_mutex.unlock();
    return probeWithLocked(default_selection.preference);
}

/// Probe a specific adapter preference without changing the process default.
/// This is what an adapter ablation uses.
pub fn probeWithAdapter(selection: AdapterSelection) ProbeResult {
    return probeCachedWith(selection);
}

/// The comptime-dispatched engine has a void API, so its GPU implementation
/// uses one process-local context.  Explicit callers/tests can still construct
/// and own a GpuContext directly.
pub fn global() !*GpuContext {
    lock(&probe_mutex);
    defer probe_mutex.unlock();

    const preferred = &probe_slots[selectionSlot(default_selection)];
    const result = if (preferred.context != null) preferred.result else probeWithLocked(default_selection);
    if (!result.available) {
        recordFallback(result.reason);
        return error.GpuError;
    }
    return &preferred.context.?;
}

/// Reset the process-local GPU state.  This is primarily useful for tests and
/// for applications that intentionally want to retry after changing their
/// driver environment; ordinary callers should rely on the one-time cache.
pub fn resetGlobal() void {
    lock(&probe_mutex);
    defer probe_mutex.unlock();

    for (&probe_slots) |*slot| {
        if (slot.context) |*context| context.deinit();
        slot.context = null;
        slot.done = false;
        slot.result = ProbeResult.unavailable(.not_probed);
    }
    probe_override_for_testing = null;
    clearFallbackReason();
}

/// Test-only dependency injection for the unavailable-device path.  A
/// successful override is intentionally not supported because a fake result
/// must not manufacture a fake `GpuContext`; tests should inject failure only.
pub fn setProbeOverrideForTesting(override: ?ProbeResult) void {
    lock(&probe_mutex);
    defer probe_mutex.unlock();

    for (&probe_slots) |*slot| {
        if (slot.context) |*context| context.deinit();
        slot.context = null;
        slot.done = false;
        slot.result = ProbeResult.unavailable(.not_probed);
    }
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

test "GPU limits bound buffers and the 2D dispatch grid" {
    const limits = GpuLimits{
        .maxComputeWorkgroupsPerDimension = 2,
        .maxStorageBufferBindingSize = 4096,
        .maxBufferSize = 8192,
    };

    try std.testing.expect(limits.canRun(4096, 4));
    try std.testing.expect(!limits.canRun(4097, 4));
    try std.testing.expect(!limits.canRun(4096, 5));
    try std.testing.expect(!limits.canRun(8193, 4));
    try std.testing.expect(!limits.canRun(0, 0));

    const available_probe = ProbeResult{
        .available = true,
        .failure = .none,
        .reason = ProbeFailure.none.reason(),
        .limits = limits,
    };
    try std.testing.expect(available_probe.canRun(4096, 4));
    try std.testing.expect(!ProbeResult.unavailable(.none).canRun(4096, 4));
}

test "adapter info keeps its own copy of the description string" {
    // `description()` returns a slice into the struct, so this asserts the copy
    // really lives inside `AdapterInfo` (a by-value receiver used to hand back a
    // dangling stack slice here).
    var info = AdapterInfo{};
    const text = "NVIDIA GeForce RTX 3050 Laptop GPU";
    @memcpy(info.description_buf[0..text.len], text);
    info.description_len = @intCast(text.len);

    const description = info.description();
    try std.testing.expectEqualStrings(text, description);
    try std.testing.expect(description.ptr == &info.description_buf);

    info.adapter_type = wgpu.WGPUAdapterType_DiscreteGPU;
    try std.testing.expect(info.isDiscrete());
    try std.testing.expectEqualStrings("discrete", info.adapterTypeName());
    info.adapter_type = wgpu.WGPUAdapterType_IntegratedGPU;
    try std.testing.expect(!info.isDiscrete());
    try std.testing.expectEqualStrings("integrated", info.adapterTypeName());
}

test "adapter classification: driver type is authoritative, names are the fallback" {
    const expect = std.testing.expect;

    // ① 驱动填了 adapterType（native/wgpu-native 会填）：直接用，不猜
    var discrete = AdapterInfo{ .adapter_type = wgpu.WGPUAdapterType_DiscreteGPU };
    try expect(discrete.kind() == .discrete);
    try expect(!discrete.isSoftware());
    try expect(std.mem.eql(u8, discrete.pathLabel(), "gpu/discrete"));

    var integrated = AdapterInfo{ .adapter_type = wgpu.WGPUAdapterType_IntegratedGPU };
    try expect(integrated.kind() == .integrated);
    try expect(std.mem.eql(u8, integrated.pathLabel(), "gpu/integrated"));

    var cpu_type = AdapterInfo{ .adapter_type = wgpu.WGPUAdapterType_CPU };
    try expect(cpu_type.kind() == .software);
    try expect(cpu_type.isSoftware());
    try expect(std.mem.eql(u8, cpu_type.pathLabel(), "gpu/software"));

    // ② 驱动没填（**浏览器实测就是这样**：nvidia/amd 都报 Unknown，只有
    //    SwiftShader 报 type=3）→ 退回到名字启发式
    var browser_sw = AdapterInfo{};
    setFixed(&browser_sw.vendor_buf, &browser_sw.vendor_len, "google");
    setFixed(&browser_sw.architecture_buf, &browser_sw.architecture_len, "swiftshader");
    try expect(browser_sw.kind() == .unknown); // 类型判不出来
    try expect(browser_sw.isSoftware()); // 但名字能判出来
    try expect(std.mem.eql(u8, browser_sw.pathLabel(), "gpu/software"));

    var browser_amd = AdapterInfo{};
    setFixed(&browser_amd.vendor_buf, &browser_amd.vendor_len, "amd");
    setFixed(&browser_amd.architecture_buf, &browser_amd.architecture_len, "gcn-5");
    try expect(!browser_amd.isSoftware()); // 不要误判硬件为软件
    try expect(std.mem.eql(u8, browser_amd.pathLabel(), "gpu/unknown"));

    var browser_nv = AdapterInfo{};
    setFixed(&browser_nv.vendor_buf, &browser_nv.vendor_len, "nvidia");
    setFixed(&browser_nv.architecture_buf, &browser_nv.architecture_len, "ampere");
    try expect(!browser_nv.isSoftware());

    // 大小写无关 + llvmpipe 这类别的软件实现
    var llvmpipe = AdapterInfo{};
    setFixed(&llvmpipe.description_buf, &llvmpipe.description_len, "llvmpipe (LLVM 22.1.8, 256 bits)");
    try expect(llvmpipe.isSoftware());

    // 选择项的标签（消融表用）
    try expect(std.mem.eql(u8, (AdapterSelection{}).label(), "auto"));
    try expect(std.mem.eql(u8, (AdapterSelection{ .preference = .high_performance }).label(), "high-performance"));
    try expect(std.mem.eql(u8, (AdapterSelection{ .force_fallback = true }).label(), "fallback(software)"));
}

fn setFixed(buf: []u8, len: *u8, text: []const u8) void {
    @memcpy(buf[0..text.len], text);
    len.* = @intCast(text.len);
}

test "adapter preference maps to the WebGPU power preference" {
    try std.testing.expectEqual(
        wgpu.WGPUPowerPreference_Undefined,
        AdapterPreference.auto.toWgpu(),
    );
    try std.testing.expectEqual(
        wgpu.WGPUPowerPreference_HighPerformance,
        AdapterPreference.high_performance.toWgpu(),
    );
    try std.testing.expectEqual(
        wgpu.WGPUPowerPreference_LowPower,
        AdapterPreference.low_power.toWgpu(),
    );
    // Each preference must land in its own cache slot, otherwise probing two
    // adapters in one process would tear down the first context.
    try std.testing.expect(AdapterPreference.auto.slot() != AdapterPreference.high_performance.slot());
    try std.testing.expect(AdapterPreference.low_power.slot() != AdapterPreference.high_performance.slot());
}

test "probeWithAdapter reports which adapter each preference selected" {
    defer resetGlobal();

    const auto = probeWithAdapter(.{ .preference = .auto });
    const fast = probeWithAdapter(.{ .preference = .high_performance });

    if (!auto.available and !fast.available) return error.SkipZigTest;

    try std.testing.expectEqual(AdapterPreference.auto, auto.preference);
    try std.testing.expectEqual(AdapterPreference.high_performance, fast.preference);

    if (auto.available) {
        try std.testing.expect(auto.adapter_info.description().len > 0);
        try std.testing.expect(auto.adapter_info.backend_type != wgpu.WGPUBackendType_Undefined);
    } else {
        // A device that cannot satisfy the default preference can still satisfy
        // an explicit one (and vice versa): the failures are cached per slot.
        try std.testing.expect(fast.failure != .not_probed);
    }

    const software = probeWithAdapter(.{ .force_fallback = true });
    std.debug.print(
        "\n[adapter] auto: {s} → {s} | high-perf: {s} → {s} | fallback: {s} → {s}\n",
        .{
            if (auto.available) auto.adapter_info.description() else auto.reason,
            if (auto.available) auto.adapter_info.pathLabel() else "-",
            if (fast.available) fast.adapter_info.description() else fast.reason,
            if (fast.available) fast.adapter_info.pathLabel() else "-",
            if (software.available) software.adapter_info.description() else software.reason,
            if (software.available) software.adapter_info.pathLabel() else "-",
        },
    );
    // 软件 fallback 槽独立缓存：探过之后不影响前两个槽
    try std.testing.expectEqual(AdapterPreference.auto, auto.preference);
    try std.testing.expectEqual(AdapterPreference.high_performance, fast.preference);
    try std.testing.expect(software.selection.force_fallback);

    // Two preferences, two live contexts: the earlier one must stay usable.
    if (auto.available and fast.available) {
        try std.testing.expect(global() != error.GpuError);
    }
}
