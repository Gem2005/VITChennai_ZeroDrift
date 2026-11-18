#!/usr/bin/env bash
# ==============================================================================
# VITChennai_ZeroDrift - Track 3 (Path Finder) verification suite
#
#   clean EuRoC / synthetic IMU -> official degrade_imu.py -> src/main.py
#   -> rule assertions (rows, heap <= 8 MB, <= 20 ms per 1 s of IMU, unit
#      quaternions, finite) -> evo_ape / evo_rpe -> plots/
#
# Thin wrapper around scripts/benchmark_all.py, run through uv.
#
#   bash scripts/evaluate.sh                       # all sequences
#   bash scripts/evaluate.sh --sequences V1_01_easy
#   bash scripts/evaluate.sh --no-plots
#
# Missing EuRoC CSVs are downloaded (IMU + ground truth only, a few MB); if that
# is not possible the EuRoC sequences are skipped and the synthetic ones still run.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v uv >/dev/null 2>&1; then
    uv sync --quiet
    exec uv run --quiet python scripts/benchmark_all.py --fetch --publish "$@"
fi

echo "[evaluate] uv not found - falling back to the current Python environment" >&2
PY="$(command -v python3 || command -v python)"
exec env ZERODRIFT_NO_UV=1 "${PY}" scripts/benchmark_all.py --fetch --publish "$@"
