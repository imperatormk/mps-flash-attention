#!/usr/bin/env bash
# Build the full wheel matrix for PyPI: one wheel per Python version (3.10-3.14),
# each containing both _C_legacy.so and _C_modern.so for cross-torch ABI support.
#
# Run from project root.
set -euo pipefail

cd "$(dirname "$0")/.."

# Wipe dist/ once at the start; subsequent per-Python builds accumulate.
rm -rf dist/
mkdir -p dist/

# (PY_BIN, LEGACY_TORCH, MODERN_TORCH) per Python minor.
# torch 2.5 doesn't ship for py3.13/3.14 — fall back to oldest available.
configs=(
    "python3.10 2.5.* 2.10.*"
    "python3.11 2.5.* 2.10.*"
    "python3.12 2.5.* 2.10.*"
    "python3.13 2.6.* 2.10.*"
    "python3.14 2.9.* 2.10.*"
)

for cfg in "${configs[@]}"; do
    set -- $cfg
    py_bin="$1"; legacy="$2"; modern="$3"
    echo ""
    echo "##############################################"
    echo "# Building for $py_bin (legacy=$legacy, modern=$modern)"
    echo "##############################################"
    PY_BIN="$py_bin" LEGACY_TORCH="$legacy" MODERN_TORCH="$modern" \
        bash scripts/build_dual_wheel.sh
done

echo ""
echo "==> All wheels:"
ls -la dist/
