# ==========================================
# STAGE 1: Builder (Compiles dependencies)
# ==========================================
FROM python:3.12-slim AS builder

# Install build dependencies (these will NOT be in the final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install 'wheel' in the builder stage only
# Then, compile all dependencies into binary .whl files
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# ==========================================
# STAGE 2: Final Runtime Image
# ==========================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Update OS packages to patch base Debian CVEs
# 2. Install ONLY libgomp1 (Required for XGBoost inference)
# Notice: No build-essential here!
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy the compiled wheels from Stage 1
COPY --from=builder /wheels /wheels

# Install the pre-compiled packages (No compilers or 'wheel' package needed!)
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application code and set ownership
COPY --chown=appuser:appuser . /app/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]