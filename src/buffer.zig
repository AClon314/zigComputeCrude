const std = @import("std");
const backend_mod = @import("backend.zig");
const BackendType = backend_mod.BackendType;

/// 统一数据容器。CPU 后端只走 cpu_ptr；GPU 后端预留 gpu_handle 与 to_device/to_host。
pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        backend: BackendType,
        cpu_ptr: []T,
        gpu_handle: u64 = 0,
        size: usize,

        pub fn init(allocator: std.mem.Allocator, backend_type: BackendType, size: usize) !Self {
            const mem = try allocator.alloc(T, size);
            return .{ .allocator = allocator, .backend = backend_type, .cpu_ptr = mem, .size = size };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.cpu_ptr);
            // GPU 后端在此释放显存（本 demo 未实现）。
        }

        /// CPU->设备。CPU 后端 no-op；GPU 后端 stub。
        pub fn toDevice(self: *Self) void {
            _ = self;
        }
        /// 设备->CPU。CPU 后端 no-op；GPU 后端 stub。
        pub fn toHost(self: *Self) void {
            _ = self;
        }
    };
}

test "buffer device buffer roundtrip" {
    const gpa = std.testing.allocator;
    var buf = try DeviceBuffer(f32).init(gpa, .cpu_simd, 16);
    defer buf.deinit();
    try std.testing.expectEqual(@as(usize, 16), buf.size);
    try std.testing.expectEqual(@as(usize, 16), buf.cpu_ptr.len);
    buf.toDevice();
    buf.toHost();
}
