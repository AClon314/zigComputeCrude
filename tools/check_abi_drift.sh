#!/usr/bin/env bash
# ABI 漂移检查：native(wgpu-native) 与 browser(emdawnwebgpu) 的 webgpu.h
# 在「compute 共用子集」上的函数原型必须逐字一致。
#
# 三类检查（符号 → 结构体布局 → 硬编码常量值）：
#   1. 函数原型：SYMBOLS 列表逐字对比；
#   2. extern struct 的字段列表：清单从 src/gpu/webgpu.zig 推导（不维护第二份），
#      字段顺序/类型/名字任一漂移都会让两端产生不同的内存布局（比函数签名更危险）；
#   3. 绑定里硬编码的常量值（如 WGPUPowerPreference_HighPerformance = 0x2）：
#      同样从绑定推导，两边头文件必须一致。
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
    wgpuComputePassEncoderDispatchWorkgroupsIndirect
    wgpuComputePassEncoderEnd wgpuDeviceCreateBuffer wgpuQueueWriteBuffer
    wgpuQueueSubmit wgpuBufferMapAsync wgpuBufferGetMappedRange wgpuBufferUnmap
    wgpuBufferGetConstMappedRange
)

proto() { # $1=header $2=symbol
    grep -h "WGPU_EXPORT.*[ *]$2(" "$1" | head -1 \
        | sed 's/ WGPU_FUNCTION_ATTRIBUTE;//' | sed 's/^WGPU_EXPORT //' | sed 's/[[:space:]]\+/ /g' | sed 's/ $//'
}

# ---------- extern struct 字段列表（从绑定推导，天然不会过期）----------
struct_fields() { # $1=header $2=type
    awk -v t="$2" '
        $0 ~ ("typedef struct " t " \\{") { inside = 1; next }
        inside && /^}/ { exit }
        !inside { next }
        {
            line = $0
            while (1) {
                if (incomment) {
                    p = index(line, "*/")
                    if (p == 0) { line = ""; break }
                    line = substr(line, p + 2); incomment = 0
                } else {
                    p = index(line, "/*")
                    if (p == 0) break
                    q = index(substr(line, p + 2), "*/")
                    if (q == 0) { line = substr(line, 1, p - 1); incomment = 1; break }
                    line = substr(line, 1, p - 1) substr(line, p + 2 + q + 2)
                }
            }
            sub(/\/\/.*/, "", line)
            gsub(/[ \t]+/, " ", line)
            sub(/^ /, "", line); sub(/ $/, "", line)
            if (line != "") print line
        }
    ' "$1"
}

bindings_structs() {
    grep -oE 'pub const WGPU[A-Za-z0-9_]* = extern struct' "$ROOT/src/gpu/webgpu.zig" \
        | sed 's/^pub const //; s/ = extern struct$//' | sort -u
}

# ---------- 绑定里硬编码的常量值 ----------
bindings_constants() {
    grep -oE 'pub const WGPU[A-Za-z0-9_]+: [A-Za-z0-9_]+ = 0x[0-9A-Fa-f]+' "$ROOT/src/gpu/webgpu.zig" \
        | sed 's/^pub const //' | sed 's/: [A-Za-z0-9_]* = / /' | tr ' ' '\t'
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

# struct 布局：字段清单必须一致（从绑定推导）
types_checked=0
while read -r t; do
    [[ -z "$t" ]] && continue
    types_checked=$((types_checked + 1))
    a="$(struct_fields "$NATIVE" "$t")"
    b="$(struct_fields "$EMDAWN" "$t")"
    if [[ -z "$b" ]]; then
        echo "STRUCT-MISSING-IN-EMDAWN  $t"; fail=1
    elif [[ -z "$a" ]]; then
        echo "STRUCT-MISSING-IN-NATIVE  $t"; fail=1
    elif [[ "$a" != "$b" ]]; then
        echo "STRUCT-DIFF               $t"
        diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") | sed 's/^/  /'
        fail=1
    fi
done < <(bindings_structs)

# 硬编码常量值：两边头文件一致（读不到的（如绑定自造的别名）两边都读不到则不报）
consts_checked=0
while IFS=$'\t' read -r name value; do
    [[ -z "$name" ]] && continue
    consts_checked=$((consts_checked + 1))
    a=$(grep -cE "\b$name = $value\b" "$NATIVE" || true)
    b=$(grep -cE "\b$name = $value\b" "$EMDAWN" || true)
    if [[ "$a" != "$b" ]]; then
        echo "CONST-DIFF                $name = $value (native=$a emdawn=$b)"
        fail=1
    fi
done < <(bindings_constants)

echo "[abi] native : $NATIVE ($(wc -l < "$NATIVE") lines${NATIVE_HAVE:+; $NATIVE_HAVE})"
echo "[abi] emdawn : $EMDAWN ($(wc -l < "$EMDAWN") lines${EMDAWN_HAVE:+; Dawn $EMDAWN_HAVE})"
if [[ $fail -eq 0 ]]; then
    echo "[abi] OK: ${#SYMBOLS[@]} compute symbols, ${types_checked} structs, ${consts_checked} constants identical"
else
    echo "[abi] FAIL: ABI drift detected" >&2
fi
exit $fail
