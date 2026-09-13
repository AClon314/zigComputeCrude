#!/usr/bin/env bash
# Assert a CPU-only consumer of the `computeAccel` module does not pull the
# WebGPU backend or the spatial module into its object file.
#
# Usage: check_tree_shake.sh <object-file>
set -euo pipefail

obj="${1:?usage: check_tree_shake.sh <object-file>}"

fail=0
check_symbols() {
  local pattern="$1"
  local count
  count=$(nm "$obj" 2>/dev/null | grep -c -- "$pattern" || true)
  if [ "$count" -ne 0 ]; then
    echo "[tree-shake] FAIL: symbol pattern '$pattern' matched $count times"
    fail=1
  fi
}
check_strings() {
  local pattern="$1"
  local count
  count=$(strings "$obj" 2>/dev/null | grep -c -- "$pattern" || true)
  if [ "$count" -ne 0 ]; then
    echo "[tree-shake] FAIL: string pattern '$pattern' matched $count times"
    fail=1
  fi
}

# GPU backend (only referenced when the consumer touches accel.gpu/runtime).
check_symbols "wgpu"
# Spatial module (S1) must not be compiled unless requested.
check_symbols "grid_hash"
check_symbols "bruteForceCounts"
# WGSL source text is embedded only when a GPU kernel is used.
check_strings "workgroup_size"

if [ "$fail" -ne 0 ]; then
  exit 1
fi

size=$(stat -c%s "$obj")
echo "[tree-shake] OK: CPU-only consumer object = ${size} bytes, no wgpu/WGSL/spatial symbols"
