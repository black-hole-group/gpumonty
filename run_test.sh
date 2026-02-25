#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
