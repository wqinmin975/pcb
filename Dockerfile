# ---------------------------------------------------------------------------
# Solderbench -- reproducible environment for the SoftwareX submission
#
# Builds an image that reproduces every number in the paper bit-for-bit:
#   RF (B6) test accuracy 0.9440 / macro-F1 0.9439 at seed 42.
#
# Build:  docker build -t solderbench:0.1.3 .
# Run:    docker run --rm solderbench:0.1.3            # version banner
#         docker run --rm solderbench:0.1.3 bench      # full benchmark
#         docker run --rm solderbench:0.1.3 test       # unit-test suite
# ---------------------------------------------------------------------------
FROM python:3.13-slim

LABEL org.opencontainers.image.title="solderbench" \
      org.opencontainers.image.description="An open-source synthetic benchmark for PCB solder-joint defect inspection algorithm development" \
      org.opencontainers.image.version="0.1.3" \
      org.opencontainers.image.source="https://github.com/wqinmin975/pcb" \
      org.opencontainers.image.url="https://doi.org/10.5281/zenodo.22959192" \
      org.opencontainers.image.licenses="MIT"

# Byte-identical runs: no .pyc, unbuffered stdout, headless Matplotlib backend.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONHASHSEED=0

WORKDIR /app

# 1) Dependencies first (layer cache), pinned in requirements.txt.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# 2) Package sources.
COPY pyproject.toml README.md LICENSE ./
COPY solderbench/ ./solderbench/
COPY examples/ ./examples/
COPY tests/ ./tests/
RUN python -m pip install --no-cache-dir --no-deps -e .

# 3) Non-root user for the shipped image.
RUN useradd -m -u 1000 solderbench && chown -R solderbench:solderbench /app
USER solderbench

# Default command: print the exact environment used for the reported numbers.
CMD ["python", "-c", "\
import matplotlib, numpy, pandas, scipy, sklearn, solderbench; \
print('solderbench   ', solderbench.__version__); \
print('python        ', __import__('sys').version.split()[0]); \
print('numpy         ', numpy.__version__); \
print('scipy         ', scipy.__version__); \
print('pandas        ', pandas.__version__); \
print('scikit-learn  ', sklearn.__version__); \
print('matplotlib    ', matplotlib.__version__)"]
