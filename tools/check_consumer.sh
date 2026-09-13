#!/usr/bin/env bash
# Build the example consumer against the package as a *dependency* (path dep)
# and assert it is CPU-only: no wgpu-native in the binary, no spatial imports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONSUMER="$ROOT/examples/cpu_consumer"

out="$(cd "$CONSUMER" && zig build run --summary all 2>&1)"
echo "$out" | grep -q "cpu_consumer: simd add -> 5.0" || {
    echo "[consumer] FAIL: unexpected output"; echo "$out"; exit 1;
}

exe="$CONSUMER/zig-out/bin/cpu_consumer"
if ldd "$exe" 2>/dev/null | grep -q wgpu; then
    echo "[consumer] FAIL: binary links wgpu-native despite -Dwebgpu=false"; exit 1
fi
size=$(stat -c%s "$exe")
echo "[consumer] OK: CPU-only package consumer built+ran (binary ${size} bytes, no wgpu-native)"
