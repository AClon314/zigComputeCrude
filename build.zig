const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // Whether to link wgpu-native.  Consumers that only use the CPU kernels can
    // pass `.webgpu = false` to `b.dependency(...)`; their build then needs no
    // wgpu-native .so/headers.  The GPU code is never *compiled* unless
    // referenced, this only removes the link step.
    const webgpu_enabled = b.option(
        bool,
        "webgpu",
        "link wgpu-native (native WebGPU backend); set false for CPU-only consumers",
    ) orelse true;
    // Where to find libwgpu_native.so + include/ (defaults to the vendored copy).
    // Published packages cannot ship vendor/, so consumers point this at their
    // own wgpu-native build.
    const wgpu_lib_dir_option = b.option(
        []const u8,
        "wgpu-lib-dir",
        "directory containing libwgpu_native.so and include/ (default: vendor/wgpu-native/lib)",
    );
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("computeAccel", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
        .link_libc = target.result.os.tag == .linux,
    });

    // T1 is the native Linux backend. Keep the WebGPU library on the module so
    // both the executable and its test artifact resolve the hand-written C ABI.
    // The absolute build-tree rpath makes `zig build test` work, while the
    // installed rpath plus copied .so makes `zig build run` relocatable under
    // zig-out/{bin,lib}.
    if (target.result.os.tag == .linux and webgpu_enabled) {
        const wgpu_lib_dir = if (wgpu_lib_dir_option) |dir|
            b.path(dir)
        else
            b.path("vendor/wgpu-native/lib");
        mod.addLibraryPath(wgpu_lib_dir);
        mod.linkSystemLibrary("wgpu_native", .{ .use_pkg_config = .no });
        mod.addRPath(wgpu_lib_dir);
        mod.addRPathSpecial("$ORIGIN/../lib");
        if (wgpu_lib_dir_option == null) {
            b.installFile("vendor/wgpu-native/lib/libwgpu_native.so", "lib/libwgpu_native.so");
        }
    }

    // Spatial primitives (S1) live in their own module: a consumer that never
    // imports it does not compile it (and it deliberately has no GPU link of
    // its own; it uses the core module's runtime when it needs the GPU).
    const spatial_mod = b.addModule("computeAccel_spatial", .{
        .root_source_file = b.path("src/spatial.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = target.result.os.tag == .linux,
        .imports = &.{
            .{ .name = "computeAccel", .module = mod },
        },
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "computeAccel",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "computeAccel" is the name you will use in your source code to
                // import this module (e.g. `@import("computeAccel")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "computeAccel", .module = mod },
            },
        }),
    });

    // The demo CLI also drives the spatial module (S1); the library modules
    // themselves stay independent (main -> both).
    exe.root_module.addImport("computeAccel_spatial", spatial_mod);

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Spatial module tests (S1).
    const spatial_tests = b.addTest(.{ .root_module = spatial_mod });
    const run_spatial_tests = b.addRunArtifact(spatial_tests);
    test_step.dependOn(&run_spatial_tests.step);

    // Tree-shake guard: a CPU-only consumer of `computeAccel` must not compile
    // the WebGPU backend nor the spatial module.  This is the build-time
    // promise behind the multi-module layout (docs/node-system-migration.md).
    const shake_mod = b.createModule(.{
        .root_source_file = b.path("tools/tree_shake_probe.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = target.result.os.tag == .linux,
        .imports = &.{
            .{ .name = "computeAccel", .module = mod },
        },
    });
    const shake_obj = b.addObject(.{ .name = "tree_shake_probe", .root_module = shake_mod });
    const shake_check = b.addSystemCommand(&.{"bash"});
    shake_check.addFileArg(b.path("tools/check_tree_shake.sh"));
    shake_check.addFileArg(shake_obj.getEmittedBin());
    test_step.dependOn(&shake_check.step);

    // Package-consumer check: the example consumer depends on this package by
    // path, imports only `computeAccel`, builds with `-Dwebgpu=false`, and must
    // run without wgpu-native.  This is the end-to-end packaging gate.
    const consumer_check = b.addSystemCommand(&.{"bash"});
    consumer_check.addFileArg(b.path("tools/check_consumer.sh"));
    test_step.dependOn(&consumer_check.step);

    const consumer_step = b.step("consumer-check", "构建并运行示例消费者（CPU-only 打包验证）");
    consumer_step.dependOn(&consumer_check.step);

    const tree_shake_step = b.step("tree-shake", "断言 CPU-only 消费者不编译 GPU/空间模块");
    tree_shake_step.dependOn(&shake_check.step);

    // ABI 漂移检查：native(wgpu-native) 与 browser(emdawnwebgpu) 的 webgpu.h
    // 在 compute 子集上必须逐字一致 —— 这是「一套绑定编两个目标」的护栏，
    // 依赖升级悄悄改签名时能当场拦住。
    // 非严格模式：emdawn 的 webgpu.h 还没解包（没跑过 zig build wasm）时 SKIP；
    // 要硬性检查用 `zig build abi-check`（缺输入即失败）。
    const abi_drift = b.addSystemCommand(&.{"bash"});
    abi_drift.addFileArg(b.path("tools/check_abi_drift.sh"));
    test_step.dependOn(&abi_drift.step);

    const abi_check_step = b.step("abi-check", "严格模式 ABI 漂移检查（缺输入即失败；依赖升级/CI 用）");
    const abi_drift_strict = b.addSystemCommand(&.{"bash"});
    abi_drift_strict.addFileArg(b.path("tools/check_abi_drift.sh"));
    abi_drift_strict.setEnvironmentVariable("ABI_DRIFT_REQUIRE", "1");
    abi_check_step.dependOn(&abi_drift_strict.step);

    // Browser WebGPU is a separate emcc link: the native module above must not
    // pull wgpu-native into the wasm object.
    addWasmStep(b, optimize);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}

// Browser WebGPU build: Zig compiles the entry point to a wasm32-freestanding
// object, then emcc supplies the Emscripten runtime and emdawnwebgpu symbols.
// This intentionally mirrors ouo's addWasmStep, without SDL3 or ASYNCIFY.
fn addWasmStep(b: *std.Build, optimize: std.builtin.OptimizeMode) void {
    const wasm_step = b.step(
        "wasm",
        "构建浏览器 WebGPU wasm: computeAccel.js + computeAccel.wasm + shell.html",
    );

    const em_cache = b.pathResolve(&.{ b.build_root.path orelse ".", ".em-cache" });

    // Force emscripten to materialize the pinned remote port before the final
    // link.  The output lives in Zig's cache rather than the source tree.
    const fetch_port = b.addSystemCommand(&.{"emcc"});
    fetch_port.setEnvironmentVariable("EM_CACHE", em_cache);
    fetch_port.addArgs(&.{
        "--use-port=deps/emdawnwebgpu.remoteport.py",
        "-c",
    });
    fetch_port.addFileArg(b.path("src/bindings/web/wasm_main.c"));
    fetch_port.addArg("-o");
    _ = fetch_port.addOutputFileArg("emdawnwebgpu-port-probe.o");

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    const wasm_webgpu_mod = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    const wasm_shader_mod = b.createModule(.{
        .root_source_file = b.path("src/gpu/shaders/add_source.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    const wasm_entry_mod = b.createModule(.{
        .root_source_file = b.path("src/abi/wasm.zig"),
        .target = wasm_target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "computeAccel_gpu_webgpu", .module = wasm_webgpu_mod },
            .{ .name = "computeAccel_add_shader", .module = wasm_shader_mod },
        },
    });
    const wasm_obj = b.addObject(.{
        .name = "computeAccel_wasm_zig",
        .root_module = wasm_entry_mod,
    });

    const emcc_cmd = b.addSystemCommand(&.{"emcc"});
    emcc_cmd.setEnvironmentVariable("EM_CACHE", em_cache);
    emcc_cmd.addArgs(&.{
        "--use-port=deps/emdawnwebgpu.remoteport.py",
        "-sDEFAULT_TO_CXX=1",
        "-sALLOW_MEMORY_GROWTH=1",
        "--closure=1",
        "-O3",
        "-sEXPORTED_FUNCTIONS=_main,_ca_wasm_status,_ca_wasm_pump,_ca_wasm_set_adapter_preference",
        "-sEXPORTED_RUNTIME_METHODS=HEAPU8",
        "-o",
    });
    const js_out = emcc_cmd.addOutputFileArg("computeAccel.js");
    emcc_cmd.addFileArg(wasm_obj.getEmittedBin());
    emcc_cmd.addFileArg(b.path("src/bindings/web/wasm_main.c"));
    emcc_cmd.step.dependOn(&wasm_obj.step);
    emcc_cmd.step.dependOn(&fetch_port.step);

    const install_dir = b.addInstallDirectory(.{
        .source_dir = js_out.dirname(),
        .install_dir = .prefix,
        .install_subdir = "webgpu",
    });
    install_dir.step.dependOn(&emcc_cmd.step);
    wasm_step.dependOn(&install_dir.step);
    wasm_step.dependOn(&b.addInstallFile(
        b.path("src/bindings/web/shell.html"),
        "webgpu/shell.html",
    ).step);
}
