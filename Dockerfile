# Multi-stage Dockerfile for Explainable Fraud Detection API
# This dockerfile creates a production-ready container with optimized layers
# and minimal attack surface.

# ========================================
# STAGE 1: Python Dependencies Builder
# ========================================
FROM python:3.11-slim as dependencies-builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python packages with optimizations
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-deps -r requirements.txt && \
    pip install -r requirements.txt

# ========================================
# STAGE 2: Production Runtime
# ========================================
FROM python:3.11-slim as production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=dependencies-builder /opt/venv /opt/venv

# Create application directory structure
RUN mkdir -p /app/src && \
    mkdir -p /app/data/raw && \
    mkdir -p /app/data/processed && \
    mkdir -p /app/models && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser requirements.txt ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/.mlflow && \
    chown -R appuser:appuser /app/.mlflow

# Switch to non-root user
USER appuser

# Set up MLflow tracking directory
ENV MLFLOW_TRACKING_URI=sqlite:////app/.mlflow/mlflow.db

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# Expose the API port
EXPOSE 8000

# Default command to run the FastAPI application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ========================================
# DEVELOPMENT STAGE (Optional)
# ========================================
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /opt/venv/bin/pip install \
    jupyter \
    ipython \
    pytest-xdist \
    pytest-cov

# Switch back to appuser
USER appuser

# Override command for development
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ========================================
# LABELS AND METADATA
# ========================================

LABEL maintainer="Vrund Patel" \
      version="1.0.0" \
      description="Explainable AI for Graph-Based Fraud Detection" \
      org.opencontainers.image.title="Fraud Detection API" \
      org.opencontainers.image.description="Graph Neural Network-based fraud detection with AI explanations" \
      org.opencontainers.image.version="1.0.0"


