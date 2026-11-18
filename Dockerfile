# ==============================================================================
# VITChennai_ZeroDrift - reproducible environment (Samsung XR Hackathon 2026,
# Track 3: Path Finder). Dependencies are installed with uv from uv.lock.
#
#   docker build -t vitchennai_zerodrift .
#   docker run --rm vitchennai_zerodrift                        # full verification suite
#   docker run --rm -v "$PWD:/io" vitchennai_zerodrift \
#       uv run python src/main.py --input /io/degraded.csv \
#       --output /io/estimated_trajectory.txt --meta /io/metadata.json
# ==============================================================================

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /workspace/VITChennai_ZeroDrift

# dependency layer (numpy for the estimator; scipy, pandas, evo, matplotlib for the suite)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project

COPY . .

# degrade -> estimate -> assert -> evo -> plots. EuRoC CSVs are fetched when the
# network allows it; otherwise the synthetic sequences are benchmarked alone.
CMD ["bash", "scripts/evaluate.sh"]
