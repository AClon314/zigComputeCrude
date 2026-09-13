const std = @import("std");
const accel = @import("computeAccel");

pub fn main() !void {
    const n = 1024;
    var out: [n]f32 = undefined;
    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    @memset(&a, 2.0);
    @memset(&b, 3.0);

    accel.ComputeEngine(.cpu_simd).add(f32, &out, &a, &b);
    std.debug.print("cpu_consumer: simd add -> {d:.1} (spatial module not imported)\n", .{out[n - 1]});
}
