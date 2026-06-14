# ==========================================
# STAGE 1: Builder (Compiles dependencies)
# ==========================================
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Keep requirements in a separate /build folder so they don't mix with wheels
WORKDIR /build
COPY requirements.txt .

# Compile all dependencies into binary .whl files and save them to /wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# ==========================================
# STAGE 2: Final Runtime Image
# ==========================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Update OS packages and install libgomp1 (Needed for XGBoost)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy ONLY the compiled wheels from Stage 1
COPY --from=builder /wheels /wheels

# Install the pre-compiled packages targeting ONLY .whl files
# Added --root-user-action=ignore to suppress the annoying pip root warning
RUN pip install --no-cache-dir --upgrade pip setuptools --root-user-action=ignore && \
    pip install --no-cache-dir /wheels/*.whl --root-user-action=ignore && \
    rm -rf /wheels

# Copy application code and set ownership
COPY --chown=appuser:appuser . /app/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]