# ==============================================================================
# VITChennai_ZeroDrift - Samsung XR Hackathon 2026, Track 3: Path Finder
# Submission image (Slide 25): Python 3.11, numpy, src/main.py as the entry point.
#
#   docker build -t vitchennai_zerodrift .
#   docker run --rm -v "$PWD:/io" vitchennai_zerodrift \
#       --input /io/imu0/data.csv --output /io/estimated_trajectory.txt --meta /io/metadata.json
#
# Everything after the image name is passed to src/main.py (flags or positional).
# The development image with the full verification suite is Dockerfile.dev.
# ==============================================================================

FROM python:3.11-slim

# one CPU thread (src/main.py sets the same variables itself before importing numpy)
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY README.md .

ENTRYPOINT ["python3", "src/main.py"]
