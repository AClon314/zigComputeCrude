#!/usr/bin/env bash
# ABI 漂移检查：native(wgpu-native) 与 browser(emdawnwebgpu) 的 webgpu.h
# 在「compute 共用子集」上的函数原型必须逐字一致。
#
# 背景：两条链路链接的是同一套 Zig 绑定（src/gpu/webgpu.zig），原理是两边的
#       webgpu.h ABI 相同。任一边升级后必须重跑本脚本，否则会出现只在某一个
#       目标上崩溃的诡异 ABI 错位。
#
# 用法：tools/check_abi_drift.sh [emdawn webgpu.h 路径]
#       不给参数时自动在 .em-cache/ports/ 下找（需先跑过一次 emcc --use-port=...）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE="$ROOT/vendor/wgpu-native/include/webgpu/webgpu.h"

if [[ $# -ge 1 ]]; then
    EMDAWN="$1"
else
    EMDAWN="$(find "$ROOT/.em-cache/ports" -path '*emdawnwebgpu_pkg/webgpu/include/webgpu/webgpu.h' 2>/dev/null | head -1)"
fi

[[ -f "$NATIVE" ]] || { echo "missing: $NATIVE (run tools/fetch_deps.sh first)" >&2; exit 2; }
[[ -n "${EMDAWN:-}" && -f "$EMDAWN" ]] || { echo "missing emdawnwebgpu webgpu.h (run: emcc --use-port=deps/emdawnwebgpu.remoteport.py:help)" >&2; exit 2; }

# compute 共用子集：绑定层实际用到的全部符号
SYMBOLS=(
    wgpuCreateInstance wgpuInstanceProcessEvents
    wgpuInstanceRequestAdapter wgpuAdapterRequestDevice wgpuAdapterGetInfo
    wgpuDeviceGetQueue wgpuDevicePushErrorScope wgpuDevicePopErrorScope
    wgpuDeviceCreateShaderModule wgpuDeviceCreateComputePipeline
    wgpuComputePipelineGetBindGroupLayout wgpuDeviceCreateBindGroupLayout
    wgpuDeviceCreateBindGroup wgpuDeviceCreatePipelineLayout
    wgpuDeviceCreateCommandEncoder wgpuCommandEncoderBeginComputePass
    wgpuCommandEncoderFinish wgpuComputePassEncoderSetPipeline
    wgpuComputePassEncoderSetBindGroup wgpuComputePassEncoderDispatchWorkgroups
    wgpuComputePassEncoderEnd wgpuDeviceCreateBuffer wgpuQueueWriteBuffer
    wgpuQueueSubmit wgpuBufferMapAsync wgpuBufferGetMappedRange wgpuBufferUnmap
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

echo "native : $NATIVE ($(wc -l < "$NATIVE") lines)"
echo "emdawn : $EMDAWN ($(wc -l < "$EMDAWN") lines)"
if [[ $fail -eq 0 ]]; then
    echo "OK: ${#SYMBOLS[@]} compute symbols identical"
else
    echo "FAIL: ABI drift detected" >&2
fi
exit $fail
