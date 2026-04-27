#!/usr/bin/env bash
# Test each (python, wheel, torch) combo.
set -u

# Map of py-version -> torch versions to try
declare -A TORCH_FOR_PY
TORCH_FOR_PY[3.10]="2.5 2.6 2.7 2.8 2.9 2.10 2.11"
TORCH_FOR_PY[3.11]="2.5 2.6 2.7 2.8 2.9 2.10 2.11"
TORCH_FOR_PY[3.12]="2.5 2.6 2.7 2.8 2.9 2.10 2.11"
TORCH_FOR_PY[3.13]="2.6 2.7 2.8 2.9 2.10 2.11"
TORCH_FOR_PY[3.14]="2.9 2.10 2.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="${DIST:-$SCRIPT_DIR/../../dist}"
TEST_PY="$SCRIPT_DIR/test_basic.py"

for py in 3.10 3.11 3.12 3.13 3.14; do
    pytag=cp${py//./}
    wheel=$(ls $DIST/*-${pytag}-*.whl 2>/dev/null | head -1)
    if [ -z "$wheel" ]; then
        echo "py$py: NO WHEEL"; continue
    fi
    for tv in ${TORCH_FOR_PY[$py]}; do
        venv=/tmp/mfa-mat-py${py}-tv${tv}
        rm -rf $venv
        python$py -m venv $venv > /dev/null 2>&1
        $venv/bin/pip install --quiet --upgrade pip > /dev/null 2>&1
        if ! $venv/bin/pip install --quiet "torch==${tv}.*" > /dev/null 2>&1; then
            echo "py$py + torch $tv: SKIP (torch install failed)"
            continue
        fi
        $venv/bin/pip install --quiet --no-deps "$wheel" > /dev/null 2>&1
        result=$($venv/bin/python "$TEST_PY" 2>&1)
        if echo "$result" | grep -q "^OK "; then
            echo "py$py + torch $tv: PASS"
        else
            err=$(echo "$result" | grep -v "Warning\|UserWarning\|Failed to init" | tail -2 | tr '\n' ' ')
            echo "py$py + torch $tv: FAIL $err"
        fi
    done
done
