//! 验证契约：把散落的 `1e-4` / `1e-5` 常量收成一张 **comptime** 表。
//!
//! 为什么要分级（而不是所有算子都用容差）：
//!
//! - 无舍入的离散量（整数、索引、布尔、f32 的 max/min/select）在数学上就是精确的，
//!   给容差只会掩盖真 bug（索引错一位、计数错一行）→ 任何精度档下都按位/值精确比较；
//! - f32 累加类（GEMM、归约和、链式管线）受 WGSL 规范**明确允许**的"重结合与融合"影响
//!   （§15.7.5 "An implementation may reassociate operations"；§17.5.32 的 `fma`
//!   精度是"继承自 `x*y+z`"，**不保证**真融合）→ 跨厂商 bit 一致不可得，只能容差，
//!   而且必须把量级写在代码里（AGENTS §1.5）。
//!
//! 开关为什么放在 **comptime**（而不是初始化/运行时/每次调用）：
//! `Precision` 是 comptime 参数，判据表在编译期折叠成常数，比较循环里**没有分支、
//! 没有额外寄存器/指令**；调用点按需实例化（同一 kernel 的 exact/tolerant/fast 各自
//! 一份比较代码）。运行期开关会在每个元素上多分支，每次调用传参会多一层参数。
//!
//! "游戏要快、科学计算要精确" 的落点：这是**判据**开关，不是 fast-math 开关。
//! 当前 kernel 集里没有 ALU 受限的算子（reduce/GEMM 实测受内存带宽与 submit/readback
//! 支配，见 README「adapter 消融」），所以放宽判据能省的是**验证**成本，
//! 不会让帧率变高；想让 f32 链路真的更快必须另开 WGSL 变体（`fast` 档预留了位置，
//! 但没有现成收益可拿，故不做预埋）。

const std = @import("std");

/// 判据严格程度。由上层在编译期选定（CLI 用 `--precision`，宿主可固定一个值）。
pub const Precision = enum {
    /// 位/值精确（f32 也要求 `==`）：只有数学上可复现的算子能过。
    /// 用在"如果这里对不上就是真 bug"的地方（离散量、单次舍入、整数归约）。
    exact,
    /// 默认：离散量精确，f32 按 `Class` 的量级给容差。
    tolerant,
    /// 放宽的 f32 判据（游戏/CI 快检）。**离散量不放松**。
    fast,

    pub fn name(self: Precision) []const u8 {
        return switch (self) {
            .exact => "exact",
            .tolerant => "tolerant",
            .fast => "fast",
        };
    }
};

/// 被比较的值属于哪一类。"类"描述的是**舍入结构**，不是数据类型。
pub const Class = enum {
    /// 无舍入的离散量：整数/索引/计数/布尔，以及 f32 的 max/min/select
    /// （只做比较与拷贝，不产生新的舍入）。
    discrete,
    /// 单次舍入的 f32 表达式：elementwise add/mul/saxpy/bias 这类结果。
    scalar,
    /// 多次舍入或顺序敏感：GEMM 点积、归约求和、链式管线的最终标量。
    accumulated,

    pub fn name(self: Class) []const u8 {
        return switch (self) {
            .discrete => "discrete",
            .scalar => "scalar",
            .accumulated => "accumulated",
        };
    }
};

pub const Tolerance = struct {
    /// 判据：`|expected - actual| <= max(absolute, relative * |expected|)`
    /// 与仓库既有写法 `@max(1.0, |reference|) * tol` **完全等价**（absolute 就是那个
    /// 近零绝对底），所以引入档位表不会放松任何既有断言。
    absolute: f64,
    relative: f64,

    pub fn isExact(self: Tolerance) bool {
        return self.absolute == 0 and self.relative == 0;
    }

    pub fn limit(self: Tolerance, scale: f64) f64 {
        if (self.isExact()) return 0;
        return @max(self.absolute, self.relative * scale);
    }

    /// `scale` 是该元素期望值的量级（单元素判据；不是整个数组的最大值——
    /// 否则小元素会被大元素抬高阈值）。
    pub fn within(self: Tolerance, diff: f64, scale: f64) bool {
        if (self.isExact()) return diff == 0;
        return diff <= self.limit(scale);
    }
};

/// 判据表。改这里 = 改全局契约，所以每一格都要有理由：
/// - tolerant/accumulated = 1e-4：README 实测 rel 1.9e-6~3.7e-6，留 ~25x 余量；
/// - tolerant/scalar = 1e-6：单次舍入，实测逐元素相等（GEMM add/bias 路径 max|diff|=0）；
/// - fast/* = 1e-3 / 1e-2：给"只看大致对"的场景，量级与 CPython `assertAlmostEqual` 同档。
pub fn tolerance(comptime precision: Precision, comptime class: Class) Tolerance {
    return switch (class) {
        .discrete => .{ .absolute = 0, .relative = 0 },
        .scalar => switch (precision) {
            .exact => .{ .absolute = 0, .relative = 0 },
            .tolerant => .{ .absolute = 1e-6, .relative = 1e-6 },
            .fast => .{ .absolute = 1e-3, .relative = 1e-3 },
        },
        .accumulated => switch (precision) {
            .exact => .{ .absolute = 0, .relative = 0 },
            .tolerant => .{ .absolute = 1e-4, .relative = 1e-4 },
            .fast => .{ .absolute = 1e-2, .relative = 1e-2 },
        },
    };
}

/// 跨厂商（不同 GPU/driver）是否承诺逐元素可复现。
/// 离散量是；含 f32 舍入的不是（WGSL §15.7.5 允许重结合/融合）。
/// 同 device+driver+kernel+grid 下，f32 类在实践中稳定，但规范不保证。
pub fn crossVendorReproducible(comptime class: Class) bool {
    return class == .discrete;
}

pub const Comparison = struct {
    max_diff: f64,
    scale: f64,
    tolerance: Tolerance,
    within: bool,

    pub fn limit(self: Comparison) f64 {
        return self.tolerance.limit(self.scale);
    }
};

/// 最大绝对差与期望值的量级（用于打印/诊断）。
/// 判据是**逐元素**的（每个元素用自己的 |expected| 当 scale，与既有 kernel 测试
/// 的写法一致），`max_diff`/`scale` 只是给人看的汇总。逐元素差异用 f64 累加，
/// 避免比较本身引入 f32 舍入。
pub fn compareF32(
    comptime precision: Precision,
    comptime class: Class,
    expected: []const f32,
    actual: []const f32,
) Comparison {
    const tol = tolerance(precision, class);
    var max_diff: f64 = 0;
    var scale: f64 = 0;
    var within = true;
    for (expected, actual) |e, a| {
        const ef: f64 = e;
        const diff = @abs(ef - @as(f64, a));
        max_diff = @max(max_diff, diff);
        scale = @max(scale, @abs(ef));
        within = within and tol.within(diff, @abs(ef));
    }
    return .{
        .max_diff = max_diff,
        .scale = scale,
        .tolerance = tol,
        .within = within,
    };
}

/// 单个标量（链式管线的最终和等）。
pub fn compareScalarF32(
    comptime precision: Precision,
    comptime class: Class,
    expected: f32,
    actual: f32,
) Comparison {
    const tol = tolerance(precision, class);
    const max_diff: f64 = @abs(@as(f64, expected) - @as(f64, actual));
    const scale: f64 = @abs(@as(f64, expected));
    return .{
        .max_diff = max_diff,
        .scale = scale,
        .tolerance = tol,
        .within = tol.within(max_diff, scale),
    };
}

/// 精确判据的通用比较：整数/布尔类型**永远**精确（任何 `Precision`/`Class` 组合都
/// 不会放松），浮点走 `Class` 对应的容差。
/// 这是"离散量不许给容差"这条不变量唯一的落地点。
pub fn expectSlices(
    comptime precision: Precision,
    comptime class: Class,
    comptime T: type,
    expected: []const T,
    actual: []const T,
) !void {
    if (expected.len != actual.len) return error.TestExpectedEqual;

    switch (@typeInfo(T)) {
        .int, .bool => try std.testing.expectEqualSlices(T, expected, actual),
        .float => try expectSlicesWithinF(tolerance(precision, class), T, expected, actual),
        else => @compileError("determinism.expectSlices: unsupported element type " ++ @typeName(T)),
    }
}

/// 单个标量的精确/容差比较（与 `expectSlices` 同一套规则）。
pub fn expectScalar(
    comptime precision: Precision,
    comptime class: Class,
    expected: anytype,
    actual: @TypeOf(expected),
) !void {
    const T = @TypeOf(expected);
    switch (@typeInfo(T)) {
        .int, .bool => {
            if (expected != actual) return error.TestExpectedEqual;
        },
        .float => try expectScalarWithinF(tolerance(precision, class), expected, actual),
        else => @compileError("determinism.expectScalar: unsupported type " ++ @typeName(T)),
    }
}

/// 单个标量的显式容差版本（与 `expectSlicesWithin` 配对）。
pub fn expectScalarWithin(
    comptime tolerance_value: Tolerance,
    expected: anytype,
    actual: @TypeOf(expected),
) !void {
    const T = @TypeOf(expected);
    switch (@typeInfo(T)) {
        .int, .bool => {
            if (expected != actual) return error.TestExpectedEqual;
        },
        .float => try expectScalarWithinF(tolerance_value, expected, actual),
        else => @compileError("determinism.expectScalarWithin: unsupported type " ++ @typeName(T)),
    }
}

fn expectScalarWithinF(tolerance_value: Tolerance, expected: anytype, actual: @TypeOf(expected)) !void {
    const ef: f64 = expected;
    const af: f64 = actual;
    if (!tolerance_value.within(@abs(ef - af), @abs(ef))) return error.TestExpectedApproxEq;
}

/// 显式容差版本：给"同一后端家族内部的更紧判据"（例如 CPU scalar vs CPU SIMD 的
/// 1e-5）用，避免为了套档位而放松既有断言。
pub fn expectSlicesWithin(
    comptime tolerance_value: Tolerance,
    comptime T: type,
    expected: []const T,
    actual: []const T,
) !void {
    if (expected.len != actual.len) return error.TestExpectedEqual;
    try expectSlicesWithinF(tolerance_value, T, expected, actual);
}

fn expectSlicesWithinF(
    tolerance_value: Tolerance,
    comptime T: type,
    expected: []const T,
    actual: []const T,
) !void {
    for (expected, actual) |e, a| {
        const ef: f64 = @floatCast(e);
        const af: f64 = @floatCast(a);
        // 逐元素判据：每个元素用自己的量级当 scale。
        if (!tolerance_value.within(@abs(ef - af), @abs(ef))) return error.TestExpectedApproxEq;
    }
}

// ---- tests ----

const expect = std.testing.expect;

test "discrete values stay exact in every precision" {
    const expected = [_]u32{ 0, 1, 2, 1024, 0xFFFF_FFFE };
    var actual = expected;
    try expectSlices(.exact, .discrete, u32, &expected, &actual);
    try expectSlices(.fast, .discrete, u32, &expected, &actual);

    // 差分一位就必须失败 —— 即便在 fast 档。
    actual[3] = 1025;
    try std.testing.expectError(
        error.TestExpectedEqual,
        expectSlices(.fast, .discrete, u32, &expected, &actual),
    );
    try std.testing.expectError(
        error.TestExpectedEqual,
        expectSlices(.fast, .scalar, u32, &expected, &actual),
    );

    try expect(tolerance(.fast, .discrete).isExact());
    try expect(crossVendorReproducible(.discrete));
    try expect(!crossVendorReproducible(.accumulated));
}

test "exact precision rejects a one-ulp float difference" {
    const expected = [_]f32{ 1.0, 0.1, 12345.678, -7.25 };
    var actual = expected;
    try expectSlices(.exact, .scalar, f32, &expected, &actual);
    try expectSlices(.exact, .accumulated, f32, &expected, &actual);

    actual[1] = @bitCast(@as(u32, @bitCast(@as(f32, 0.1))) + 1); // 1 ulp
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectSlices(.exact, .scalar, f32, &expected, &actual),
    );
    // 而 tolerant 档接受它（单次舍入类实测就是逐元素相等，1 ulp 仍远小于 1e-6 相对量级）。
    try expectSlices(.tolerant, .scalar, f32, &expected, &actual);
}

test "tolerant and fast differ by orders of magnitude, and both bracket the measured GPU error" {
    const expected = [_]f32{1.0};
    const measured_abs = [_]f32{1.0 + 3.7e-6}; // README 异质链实测 rel 3.7e-6
    const loose_abs = [_]f32{1.0 + 5e-3};
    const broken_abs = [_]f32{1.0 + 1.0};

    // 实测误差：tolerant 过，exact 不过。
    try expectSlices(.tolerant, .accumulated, f32, &expected, &measured_abs);
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectSlices(.exact, .accumulated, f32, &expected, &measured_abs),
    );

    // fast 比 tolerant 宽 100 倍，但仍能挡住 1.0 这种真错。
    try expect(tolerance(.fast, .accumulated).limit(1.0) > tolerance(.tolerant, .accumulated).limit(1.0) * 10);
    try expectSlices(.fast, .accumulated, f32, &expected, &loose_abs);
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectSlices(.fast, .accumulated, f32, &expected, &broken_abs),
    );
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectSlices(.tolerant, .accumulated, f32, &expected, &loose_abs),
    );
}

test "near-zero results are covered by the absolute floor, not the scale" {
    // scale = max|expected| = 0 → 相对项失效，此时全靠 absolute。
    const expected = [_]f32{0.0};
    const tiny = [_]f32{1e-7};
    const not_tiny = [_]f32{1e-3};
    try expectSlices(.tolerant, .accumulated, f32, &expected, &tiny);
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectSlices(.tolerant, .accumulated, f32, &expected, &not_tiny),
    );
}

test "explicit tolerance keeps a tighter internal gate" {
    const expected = [_]f32{1.0};
    const off_by_tiny = [_]f32{1.0 + 5e-6};
    // CPU scalar vs CPU SIMD 的既有判据是 1e-5，比 tolerant/accumulated 更紧。
    try expectSlicesWithin(.{ .absolute = 1e-5, .relative = 1e-5 }, f32, &expected, &off_by_tiny);
    const off_by_loose = [_]f32{1.0 + 5e-5};
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectSlicesWithin(.{ .absolute = 1e-5, .relative = 1e-5 }, f32, &expected, &off_by_loose),
    );
}

test "scalar comparison and length mismatch" {
    try expectScalar(.tolerant, .accumulated, @as(f32, 75691450.0), 75691450.0);
    try expectScalar(.tolerant, .accumulated, @as(f32, 1.0), 1.0 + 1e-6);
    try std.testing.expectError(
        error.TestExpectedApproxEq,
        expectScalar(.exact, .accumulated, @as(f32, 1.0), 1.0 + 1e-6),
    );
    try std.testing.expectError(
        error.TestExpectedEqual,
        expectSlices(.tolerant, .discrete, u32, &[_]u32{ 1, 2 }, &[_]u32{1}),
    );

    const comparison = compareScalarF32(.tolerant, .accumulated, 75691450.0, 75691170.0);
    try expect(comparison.within);
    try expect(comparison.max_diff == 280.0);
}
