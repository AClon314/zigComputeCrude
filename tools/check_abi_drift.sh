#!/usr/bin/env bash
# ABI 漂移检查：native(wgpu-native) 与 browser(emdawnwebgpu) 的 webgpu.h
# 在「compute 共用子集」上的函数原型必须逐字一致。
#
# 背景：两条链路链接的是同一套 Zig 绑定（src/gpu/webgpu.zig），原理是两边的
#       webgpu.h ABI 相同。任一边升级后如果签名发生漂移，就会出现「只在某一个
#       目标上崩溃/静默出错」的诡异问题，所以这个检查接进了 `zig build test`。
#
# 用法：
#   tools/check_abi_drift.sh [--require] [emdawn 的 webgpu.h 路径]
#
#   --require           严格模式：输入缺失即报错（CI / 依赖升级时用）
#   环境变量 ABI_DRIFT_REQUIRE=1 等价于 --require
#
# 退出码：
#   0  OK，或（非严格模式下）输入缺失而 SKIP
#   1  检测到 ABI 漂移
#   2  严格模式下输入缺失
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE="$ROOT/vendor/wgpu-native/include/webgpu/webgpu.h"

REQUIRE="${ABI_DRIFT_REQUIRE:-0}"
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --require) REQUIRE=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [[ ${#ARGS[@]} -ge 1 ]]; then
    EMDAWN="${ARGS[0]}"
else
    EMDAWN="$(find "$ROOT/.em-cache/ports" -path '*emdawnwebgpu_pkg/webgpu/include/webgpu/webgpu.h' 2>/dev/null | head -1)"
fi

# ---------- 输入缺失处理（非严格模式 SKIP，严格模式失败）----------
missing=""
[[ -f "$NATIVE" ]] || missing="native 头文件: $NATIVE（先跑 tools/fetch_deps.sh）"
if [[ -z "$missing" && ( -z "${EMDAWN:-}" || ! -f "$EMDAWN" ) ]]; then
    missing="emdawnwebgpu 头文件（.em-cache 未解包；跑一次 zig build wasm，或 emcc --use-port=deps/emdawnwebgpu.remoteport.py:help）"
fi

if [[ -n "$missing" ]]; then
    if [[ "$REQUIRE" == "1" ]]; then
        echo "[abi] FAIL: 缺少 $missing" >&2
        exit 2
    fi
    echo "[abi] SKIP: 缺少 $missing"
    echo "[abi]       （想让它成为硬性检查：ABI_DRIFT_REQUIRE=1 或 zig build abi-check）"
    exit 0
fi

# ---------- 版本提示（pins.env 是唯一事实来源，不一致只 WARN）----------
pins_version() { # $1=变量名
    [[ -f "$ROOT/deps/pins.env" ]] || return 0
    sed -n "s/^$1=\"\{0,1\}\([^\"]*\)\"\{0,1\}$/\1/p" "$ROOT/deps/pins.env" | head -1
}

NATIVE_PIN="$(pins_version WGPU_NATIVE_VERSION)"
STAMP="$ROOT/vendor/wgpu-native/.pinned-version"
NATIVE_HAVE="$(cat "$STAMP" 2>/dev/null || cat "$ROOT/vendor/wgpu-native/wgpu-native-meta/wgpu-native-git-tag" 2>/dev/null || true)"

EMDAWN_PIN="$(pins_version EMDAWNWEBGPU_VERSION)"
EMDAWN_HAVE="$(sed -n 's/.*\(v[0-9]\{8\}\.[0-9]\{6\}\).*/\1/p' "$(dirname "$EMDAWN")/../../../VERSION.txt" 2>/dev/null | head -1)"

if [[ -n "$NATIVE_PIN" && -n "$NATIVE_HAVE" && "$NATIVE_PIN" != "$NATIVE_HAVE" ]]; then
    echo "[abi] WARN: wgpu-native 版本与 pins.env 不一致: pins=$NATIVE_PIN 本地=$NATIVE_HAVE"
fi
if [[ -n "$EMDAWN_PIN" && -n "$EMDAWN_HAVE" && "$EMDAWN_PIN" != "$EMDAWN_HAVE" ]]; then
    echo "[abi] WARN: emdawnwebgpu 版本与 pins.env 不一致: pins=$EMDAWN_PIN 解包=$EMDAWN_HAVE"
    echo "[abi]       （升级依赖后需清掉 .em-cache/ports/emdawnwebgpu.remoteport* 重新解包）"
fi

# ---------- compute 共用子集：绑定层实际用到的全部符号 ----------
SYMBOLS=(
    wgpuCreateInstance wgpuInstanceProcessEvents
    wgpuInstanceRequestAdapter wgpuAdapterRequestDevice wgpuAdapterGetInfo
    wgpuAdapterGetLimits
    wgpuDeviceGetQueue wgpuDeviceGetLimits wgpuDevicePushErrorScope wgpuDevicePopErrorScope
    wgpuDeviceCreateShaderModule wgpuDeviceCreateComputePipeline
    wgpuComputePipelineGetBindGroupLayout wgpuDeviceCreateBindGroupLayout
    wgpuDeviceCreateBindGroup wgpuDeviceCreatePipelineLayout
    wgpuDeviceCreateCommandEncoder wgpuCommandEncoderBeginComputePass
    wgpuCommandEncoderCopyBufferToBuffer wgpuCommandEncoderFinish
    wgpuComputePassEncoderSetPipeline
    wgpuComputePassEncoderSetBindGroup wgpuComputePassEncoderDispatchWorkgroups
    wgpuComputePassEncoderEnd wgpuDeviceCreateBuffer wgpuQueueWriteBuffer
    wgpuQueueSubmit wgpuBufferMapAsync wgpuBufferGetMappedRange wgpuBufferUnmap
    wgpuBufferGetConstMappedRange
)

proto() { # $1=header $2=symbol
    grep -h "WGPU_EXPORT.*[ *]$2(" "$1" | head -1 \
        | sed 's/ WGPU_FUNCTION_ATTRIBUTE;//' | sed 's/^WGPU_EXPORT //' | sed 's/[[:space:]]\+/ /g' | sed 's/ $//'
}

fail=0
for s in "${SYMBOLS[@]}"; do
    a="$(proto "$NATIVE" "$s")"
    b="$(proto "$EMDAWN" "$s")"
    if [[ -z "$b" ]]; then
        echo "MISSING-IN-EMDAWN  $s"; fail=1
    elif [[ -z "$a" ]]; then
        echo "MISSING-IN-NATIVE  $s"; fail=1
    elif [[ "$a" != "$b" ]]; then
        echo "DIFF               $s"
        echo "  native: $a"
        echo "  emdawn: $b"
        fail=1
    fi
done

echo "[abi] native : $NATIVE ($(wc -l < "$NATIVE") lines${NATIVE_HAVE:+; $NATIVE_HAVE})"
echo "[abi] emdawn : $EMDAWN ($(wc -l < "$EMDAWN") lines${EMDAWN_HAVE:+; Dawn $EMDAWN_HAVE})"
if [[ $fail -eq 0 ]]; then
    echo "[abi] OK: ${#SYMBOLS[@]} compute symbols identical"
else
    echo "[abi] FAIL: ABI drift detected" >&2
fi
exit $fail
