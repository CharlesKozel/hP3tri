#!/usr/bin/env bash
set -euo pipefail

# Auto-detect GPU and set Taichi backend
if [ -z "${TAICHI_ARCH:-}" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        export TAICHI_ARCH=cuda
    else
        echo "No GPU detected, falling back to CPU"
        export TAICHI_ARCH=cpu
    fi
fi
echo "Taichi backend: $TAICHI_ARCH"

# Find Jep native library path without importing jep (which requires Java embedding)
JEP_PATH=$(python3 -c "import importlib.util; print(importlib.util.find_spec('jep').submodule_search_locations[0])")
echo "Jep path: $JEP_PATH"

export JAVA_OPTS="${JAVA_OPTS:-} \
    -Djava.library.path=$JEP_PATH \
    -Xms${JVM_XMS:-2g} \
    -Xmx${JVM_XMX:-8g}"

export PYTHONPATH=/app/python
export LD_PRELOAD=${LD_PRELOAD:-libpython3.12.so}

echo "Starting hP3tri server..."
exec /app/kotlin-dist/bin/hp3tri "$@"
