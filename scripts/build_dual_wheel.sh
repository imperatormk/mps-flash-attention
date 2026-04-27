#!/usr/bin/env bash
# Build a dual-ABI wheel for ONE Python version.
# Each wheel contains _C_legacy.so (built against torch X.Y, ABI-compatible
# with torch 2.5-2.9) AND _C_modern.so (built against torch 2.10, ABI-compatible
# with 2.10+). The package's __init__.py picks the right one at import time
# based on the user's installed torch version.
#
# Usage:
#   PY_BIN=python3.12 LEGACY_TORCH="2.5.*" MODERN_TORCH="2.10.*" \
#     bash scripts/build_dual_wheel.sh
#
# If LEGACY_TORCH is "skip", only the modern .so is built (use for py3.14
# where torch 2.5 isn't available).
set -euo pipefail

PY_BIN="${PY_BIN:-python3.12}"
LEGACY_TORCH="${LEGACY_TORCH:-2.5.*}"
MODERN_TORCH="${MODERN_TORCH:-2.10.*}"

cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY_TAG=$($PY_BIN -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
WORK=/tmp/mfa-dualbuild-$PY_TAG
LEGACY_VENV="$WORK/legacy"
MODERN_VENV="$WORK/modern"

# Clean per-Python state but keep dist/ to accumulate wheels.
rm -rf "$WORK"
mkdir -p "$WORK"
rm -rf build/ mps_flash_attn.egg-info
rm -f mps_flash_attn/_C_legacy.cpython-*.so
rm -f mps_flash_attn/_C_modern.cpython-*.so
rm -f mps_flash_attn/_C_legacy.so
rm -f mps_flash_attn/_C_modern.so

# Build Swift bridge once (idempotent across python versions).
if [ ! -f swift-bridge/.build/release/libMFABridge.dylib ]; then
    echo "==> Building Swift bridge"
    (cd swift-bridge && swift build -c release > /dev/null)
fi

mk_venv() {
    local dir="$1"; local torch_spec="$2"
    "$PY_BIN" -m venv "$dir"
    "$dir/bin/pip" install --quiet --upgrade pip
    "$dir/bin/pip" install --quiet "torch==$torch_spec" wheel setuptools build > /dev/null
    "$dir/bin/python" -c "import torch; print(f'  $dir torch:', torch.__version__)"
}

build_one() {
    local venv="$1"; local ext_name="$2"
    echo "==> Building $ext_name ($PY_TAG) using $(basename $venv)"
    MFA_EXT_NAME="$ext_name" "$venv/bin/python" setup.py build_ext --inplace \
        > "$WORK/${ext_name}.log" 2>&1 || {
        tail -20 "$WORK/${ext_name}.log"
        echo "build_ext failed for $ext_name"; exit 1;
    }
}

# Legacy build (skipped if not available for this Python version).
if [ "$LEGACY_TORCH" != "skip" ]; then
    echo "==> Setting up legacy venv (torch $LEGACY_TORCH)"
    mk_venv "$LEGACY_VENV" "$LEGACY_TORCH"
    build_one "$LEGACY_VENV" "_C_legacy"
fi

echo "==> Setting up modern venv (torch $MODERN_TORCH)"
mk_venv "$MODERN_VENV" "$MODERN_TORCH"
build_one "$MODERN_VENV" "_C_modern"

echo "==> Built .so files:"
ls -la mps_flash_attn/_C_*.so

PYVER=$($PY_BIN -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
echo "==> Packaging wheel for $PY_TAG ($PYVER)"
MFA_SKIP_EXT=1 "$MODERN_VENV/bin/python" setup.py bdist_wheel \
    --plat-name macosx_11_0_arm64 \
    --python-tag "$PYVER" \
    > "$WORK/wheel.log" 2>&1 || {
    tail -30 "$WORK/wheel.log"
    echo "bdist_wheel failed"; exit 1;
}

# The wheel was built without ext_modules so setuptools tagged it as purelib
# with abi=none. Fix the WHEEL metadata + filename to reflect the truth: this
# is a platlib (contains native .so) with cp3X abi. We also rename the file
# from cp3X-none-... to cp3X-cp3X-... so PyPI/pip see a strict abi tag.
echo "==> Fixing wheel metadata"
WHEEL_FILE=$(ls dist/mps_flash_attn-*-${PYVER}-none-macosx_11_0_arm64.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: built wheel not found in dist/"
    exit 1
fi
"$MODERN_VENV/bin/python" - <<PY
import zipfile, re, hashlib, base64
from pathlib import Path

src = Path("$WHEEL_FILE")
pyver = "$PYVER"
new_name = src.name.replace(f"-{pyver}-none-", f"-{pyver}-{pyver}-")
dst = src.parent / new_name

# Read everything from the source wheel into memory.
with zipfile.ZipFile(src, "r") as zin:
    items = [(item, zin.read(item.filename)) for item in zin.infolist()]

# Patch the WHEEL file (metadata) and recompute its RECORD entry.
def b64sha256(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

new_wheel_data = None
wheel_path = None
for i, (item, data) in enumerate(items):
    if item.filename.endswith(".dist-info/WHEEL"):
        text = data.decode()
        text = re.sub(r"^Root-Is-Purelib:.*\$", "Root-Is-Purelib: false", text, flags=re.M)
        text = re.sub(r"^Tag: .*\$", f"Tag: {pyver}-{pyver}-macosx_11_0_arm64", text, flags=re.M)
        new_wheel_data = text.encode()
        wheel_path = item.filename
        items[i] = (item, new_wheel_data)
        break

# Rewrite RECORD so the WHEEL row reflects the new content.
for i, (item, data) in enumerate(items):
    if item.filename.endswith(".dist-info/RECORD"):
        new_lines = []
        for ln in data.decode().splitlines():
            if ln.startswith(wheel_path + ","):
                digest = b64sha256(new_wheel_data)
                new_lines.append(f"{wheel_path},sha256={digest},{len(new_wheel_data)}")
            else:
                new_lines.append(ln)
        items[i] = (item, ("\n".join(new_lines) + "\n").encode())
        break

with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    for item, data in items:
        zout.writestr(item, data)

src.unlink()
print(f"  -> {dst.name}")
PY

echo "==> dist/:"
ls -la dist/
