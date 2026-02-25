#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] Script failed at line ${LINENO} (exit code $?)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Preflight checks ─────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    echo "[ERROR] Python is not installed or not on PATH." >&2
    exit 1
fi

MISSING_LIBS=()
for lib in numpy pandas matplotlib scipy; do
    python -c "import ${lib}" &>/dev/null || MISSING_LIBS+=("${lib}")
done

if [[ ${#MISSING_LIBS[@]} -gt 0 ]]; then
    echo "[ERROR] The following required Python libraries are not installed: ${MISSING_LIBS[*]}" >&2
    echo "        Install them with: pip install ${MISSING_LIBS[*]}" >&2
    exit 1
fi

# ── Step 1: detect CPU cores ─────────────────────────────────────────────────
NCORES=$(nproc)
echo "[1/4] Detected ${NCORES} CPU cores"

# ── Step 2: compile ──────────────────────────────────────────────────────────
echo "[2/4] Compiling with make -j${NCORES}"
make -j"${NCORES}"

# ── Step 3: run simulation ───────────────────────────────────────────────────
echo "[3/4] Running: ./gpumonty -par test.par"
./gpumonty -par test.par

# ── Step 4: compare spectra ──────────────────────────────────────────────────
echo "[4/4] Running comparison (compare.py)"
cd python
python compare.py
