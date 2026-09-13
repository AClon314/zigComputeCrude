#!/usr/bin/env bash
# 拉取 computeAccel 的外部依赖（幂等，可重复执行）。
#
# 依赖清单与版本见 deps/pins.env（唯一事实来源），本脚本只按 pin 下载。
# wgpu-native 预编译包较大(~16MB zip)，故不入库：放在 vendor/（.gitignore 忽略）。
# emdawnwebgpu 的 "remote port" 只有一个 7KB 的 .py（已入库 deps/），
# 真实包由 emcc 按需下载到 .em-cache/（同样不入库）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../deps/pins.env
source "$ROOT/deps/pins.env"

DEST="$ROOT/vendor/wgpu-native"
STAMP="$DEST/.pinned-version"

if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$WGPU_NATIVE_VERSION" && -f "$DEST/lib/libwgpu_native.so" ]]; then
    echo "wgpu-native $WGPU_NATIVE_VERSION: already vendored ($DEST)"
    exit 0
fi

echo "fetching wgpu-native $WGPU_NATIVE_VERSION ($WGPU_NATIVE_ASSET)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -LfsS --retry 3 -o "$TMP/wgpu-native.zip" "$WGPU_NATIVE_URL"

rm -rf "$DEST"
mkdir -p "$DEST"
unzip -q "$TMP/wgpu-native.zip" -d "$DEST"
echo "$WGPU_NATIVE_VERSION" > "$STAMP"
echo "done: $(find "$DEST" -type f | wc -l) files in $DEST"
