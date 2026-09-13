//! Element-wise kernels (`add`, `saxpy`) on the M0 runtime.
//!
//! This replaces the original `gpu/pipeline.zig` implementation, which carried
//! its own kernel enum, per-kernel caches inside `GpuContext` and a bespoke
//! dispatch/readback path.  The runtime already provides all of that, so this
//! module is just: one `Runner` per kernel (kernels + resident buffers + bind
//! group, keyed by device/size) plus the comptime-engine entry points.
//!
//! Semantics are unchanged: every call uploads its inputs, dispatches, and
//! reads the result back (the "per_call" shape).  Residency/chaining is the
//! caller's choice through `runtime.Buffer`/`Chain` APIs.

const std = @import("std");
const runtime = @import("../runtime.zig");
const context_mod = @import("../gpu/context.zig");

const add_shader = @embedFile("../gpu/shaders/add.wgsl");
const saxpy_shader = @embedFile("../gpu/shaders/saxpy.wgsl");

pub const Kind = enum { add, saxpy };

const Params = extern struct {
    alpha: f32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const Runner = struct {
    ctx: *runtime.Device,
    byte_size: usize,
    kind: Kind,

    a: runtime.Buffer,
    b: runtime.Buffer,
    out: runtime.Buffer,
    params: runtime.Buffer,
    kernel: runtime.Kernel,
    bind: ?*anyopaque,

    fn init(ctx: *runtime.Device, kind: Kind, byte_size: usize) !Runner {
        var self = Runner{
            .ctx = ctx,
            .byte_size = byte_size,
            .kind = kind,
            .a = undefined,
            .b = undefined,
            .out = undefined,
            .params = undefined,
            .kernel = undefined,
            .bind = null,
        };

        self.a = try runtime.Buffer.init(ctx, byte_size, runtime.buffer.storage_r);
        errdefer self.a.deinit(ctx);
        self.b = try runtime.Buffer.init(ctx, byte_size, runtime.buffer.storage_r);
        errdefer self.b.deinit(ctx);
        self.out = try runtime.Buffer.init(ctx, byte_size, runtime.buffer.storage_rw);
        errdefer self.out.deinit(ctx);
        self.params = try runtime.Buffer.init(ctx, 16, runtime.buffer.uniform);
        errdefer self.params.deinit(ctx);

        const add_bindings = [_]runtime.Binding{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
        };
        const saxpy_bindings = [_]runtime.Binding{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        };
        const bindings: []const runtime.Binding = switch (kind) {
            .add => &add_bindings,
            .saxpy => &saxpy_bindings,
        };
        const shader = switch (kind) {
            .add => add_shader,
            .saxpy => saxpy_shader,
        };
        self.kernel = try runtime.Kernel.init(ctx, shader, "main", bindings, 64);
        errdefer self.kernel.deinit();

        self.bind = switch (kind) {
            .add => try self.kernel.createBindGroup(&.{ &self.a, &self.b, &self.out }),
            .saxpy => try self.kernel.createBindGroup(&.{ &self.a, &self.b, &self.out, &self.params }),
        };
        return self;
    }

    fn deinit(self: *Runner) void {
        if (self.bind) |handle| runtime.releaseBindGroup(handle);
        self.bind = null;
        self.kernel.deinit();
        self.params.deinit(self.ctx);
        self.out.deinit(self.ctx);
        self.b.deinit(self.ctx);
        self.a.deinit(self.ctx);
    }
};

var runners: [2]?Runner = .{ null, null };
var runner_mutex: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!runner_mutex.tryLock()) std.atomic.spinLoopHint();
}

fn unlock() void {
    runner_mutex.unlock();
}

/// Drop the cached kernels/buffers (tests; also useful after a device change).
pub fn resetCache() void {
    lock();
    defer unlock();
    for (&runners) |*slot| {
        if (slot.*) |*runner| runner.deinit();
        slot.* = null;
    }
}

fn ensureRunner(ctx: *runtime.Device, kind: Kind, byte_size: usize) !*Runner {
    const index = @intFromEnum(kind);
    lock();
    defer unlock();

    if (runners[index]) |*runner| {
        if (runner.ctx.device == ctx.device and runner.byte_size == byte_size) return runner;
        runner.deinit();
        runners[index] = null;
    }
    runners[index] = try Runner.init(ctx, kind, byte_size);
    return &runners[index].?;
}

fn execute(
    ctx: *runtime.Device,
    kind: Kind,
    out: []f32,
    x: []const f32,
    y: []const f32,
    alpha: ?f32,
    iterations: usize,
) !void {
    if (out.len != x.len or x.len != y.len) return error.GpuError;
    if (out.len == 0 or iterations == 0) return error.GpuError;
    if (out.len > std.math.maxInt(usize) / @sizeOf(f32)) return error.GpuError;
    const byte_size = out.len * @sizeOf(f32);
    const groups = (out.len + 63) / 64;
    if (!ctx.limits.canRun(byte_size, groups)) return error.GpuError;

    const runner = try ensureRunner(ctx, kind, byte_size);
    runner.a.markHostDirty();
    try runner.a.toDevice(ctx, std.mem.sliceAsBytes(x));
    runner.b.markHostDirty();
    try runner.b.toDevice(ctx, std.mem.sliceAsBytes(y));
    if (kind == .saxpy) {
        var params = Params{ .alpha = alpha orelse return error.GpuError };
        runner.params.markHostDirty();
        try runner.params.toDevice(ctx, std.mem.asBytes(&params));
    }

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    const grid = try runner.kernel.gridLinear(out.len);
    // Built here (not returned from a helper) so the pointer array cannot
    // outlive the stack frame that owns it.
    const all_buffers = [4]*runtime.Buffer{ &runner.a, &runner.b, &runner.out, &runner.params };
    const buffers = if (kind == .add) all_buffers[0..3] else all_buffers[0..4];
    for (0..iterations) |_| {
        try chain.dispatch(&runner.kernel, runner.bind.?, buffers, grid);
    }
    try chain.download(&runner.out, std.mem.sliceAsBytes(out));
    try chain.submit();
}

pub fn addWithContext(
    ctx: *context_mod.GpuContext,
    out: []f32,
    a: []const f32,
    b: []const f32,
) !void {
    return execute(ctx, .add, out, a, b, null, 1);
}

pub fn saxpyWithContext(
    ctx: *context_mod.GpuContext,
    alpha: f32,
    out: []f32,
    x: []const f32,
    y: []const f32,
) !void {
    return execute(ctx, .saxpy, out, x, y, alpha, 1);
}

pub fn addBatchedWithContext(
    ctx: *context_mod.GpuContext,
    out: []f32,
    a: []const f32,
    b: []const f32,
    iterations: usize,
) !void {
    return execute(ctx, .add, out, a, b, null, iterations);
}

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
    const ctx = context_mod.global() catch |err| {
        if (lastFallbackReason() == null) recordFallback(@errorName(err));
        return err;
    };
    addWithContext(ctx, out, a, b) catch |err| {
        recordFallback(@errorName(err));
        return err;
    };
}

pub fn saxpy(alpha: f32, out: []f32, x: []const f32, y: []const f32) !void {
    clearFallbackReason();
    const ctx = context_mod.global() catch |err| {
        if (lastFallbackReason() == null) recordFallback(@errorName(err));
        return err;
    };
    saxpyWithContext(ctx, alpha, out, x, y) catch |err| {
        recordFallback(@errorName(err));
        return err;
    };
}

pub fn addBatched(out: []f32, a: []const f32, b: []const f32, iterations: usize) !void {
    clearFallbackReason();
    const ctx = context_mod.global() catch |err| {
        if (lastFallbackReason() == null) recordFallback(@errorName(err));
        return err;
    };
    addBatchedWithContext(ctx, out, a, b, iterations) catch |err| {
        recordFallback(@errorName(err));
        return err;
    };
}
