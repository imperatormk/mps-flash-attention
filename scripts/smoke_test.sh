#!/usr/bin/env bash
# Smoke test: create a fresh venv, install mps-flash-attn from PyPI plus a
# specified torch version, and run a one-shot flash_attention call on MPS.
#
# Usage:
#   bash scripts/smoke_test.sh                    # py3.12, torch 2.11 (defaults)
#   PY=3.13 TORCH=2.10 bash scripts/smoke_test.sh
#
# Prints PASS or FAIL and the system info that matters for triage (chip, OS).
set -u

if [ -n "${PY:-}" ]; then
    PY_BIN="python$PY"
else
    # Default: whatever python3 resolves to. Fall back to common minor
    # versions if `python3` itself doesn't exist.
    for cand in python3 python3.12 python3.13 python3.11 python3.10 python3.14; do
        if command -v "$cand" >/dev/null 2>&1; then
            PY_BIN="$cand"; break
        fi
    done
fi
if ! command -v "${PY_BIN:-}" >/dev/null 2>&1; then
    echo "FAIL: no python3 found on PATH"; exit 1
fi
PY=$("$PY_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PY" in
    3.10|3.11|3.12|3.13|3.14) ;;
    *)
        echo "FAIL: mps-flash-attn ships wheels for Python 3.10-3.14, not $PY."
        echo "      Install one of those (e.g. brew install python@3.12) and retry."
        exit 1 ;;
esac
TORCH="${TORCH:-2.11}"
VENV="${VENV:-/tmp/mfa-smoke-py${PY}-pt${TORCH}}"

echo "=== System ==="
sw_vers 2>/dev/null | sed 's/^/  /'
echo "  chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo ""

echo "=== Setup ($VENV) ==="
# Run from /tmp so we never accidentally import the in-tree source instead
# of the installed package.
cd /tmp
rm -rf "$VENV"
if ! "$PY_BIN" -m venv "$VENV" 2>/dev/null; then
    echo "FAIL: $PY_BIN -m venv failed"; exit 1
fi
"$VENV/bin/pip" install --quiet --upgrade pip
echo "Installing torch==$TORCH.* and mps-flash-attn..."
if ! "$VENV/bin/pip" install --quiet "torch==$TORCH.*" mps-flash-attn 2>&1 | tail -5; then
    echo "FAIL: pip install failed"; exit 1
fi

INSTALLED_TORCH=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1)
INSTALLED_MFA=$("$VENV/bin/python" -c 'from importlib.metadata import version; print(version("mps-flash-attn"))' 2>/dev/null | tail -1)
echo "  torch:           $INSTALLED_TORCH"
echo "  mps-flash-attn:  $INSTALLED_MFA"
echo ""

echo "=== Run ==="
"$VENV/bin/python" - <<'PY' 2>&1
import sys
import torch
from mps_flash_attn import flash_attention

if not torch.backends.mps.is_available():
    print("FAIL: MPS not available on this machine")
    sys.exit(1)

q = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)

try:
    out = flash_attention(q, k, v)
    torch.mps.synchronize()
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

if out.shape != (2, 8, 4096, 64) or out.dtype != torch.float16:
    print(f"FAIL: bad output shape/dtype: {out.shape} {out.dtype}")
    sys.exit(1)

# Also do a quick sanity check vs SDPA on CPU (small slice).
qs, ks, vs = (t[:1, :1, :64, :].cpu().float() for t in (q, k, v))
ref = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
got = out[:1, :1, :64, :].cpu().float()
err = (ref - got).abs().max().item()
if err > 0.5:
    print(f"WARN: numeric drift max-abs={err:.3f} (high but probably ok for fp16)")

print(f"PASS  shape={tuple(out.shape)}  fp16  max_err_vs_sdpa={err:.4f}")
PY
EXIT=$?

echo ""
if [ $EXIT -eq 0 ]; then
    echo "===> SUCCESS"
else
    echo "===> FAILED (exit $EXIT)"
fi
exit $EXIT
