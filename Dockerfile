# =============================================================================
# APGI Framework Dockerfile - Multi-stage build for GUI (Python 3.9)
# =============================================================================
# Build stage
FROM python:3.9-slim AS gui-builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage for GUI
FROM python:3.9-slim AS gui

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=gui-builder /opt/venv /opt/venv

# Create non-root user
RUN useradd --create-home --shell /bin/bash apgi

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=apgi:apgi . .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data results logs figures reports session_state apgi_outputs/{dashboard,exports,figures,reports} && \
    chown -R apgi:apgi /app

# Switch to non-root user
USER apgi

# Expose port for web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import apgi_framework; print('OK')" || exit 1

# Default command for GUI
CMD ["python", "GUI-Launcher.py"]

# =============================================================================
# APGI REST API Dockerfile - Multi-stage build for API (Python 3.11)
# =============================================================================
# Build stage
FROM python:3.11-slim AS api-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage for API
FROM python:3.11-slim AS api

WORKDIR /app

# Install runtime dependencies
# libtk8.6: apgi_framework imports tkinter at package import, even for the API
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=api-builder /root/.local /root/.local

# Copy application code
COPY api/ ./api/
COPY apgi_framework/ ./apgi_framework/
COPY config/ ./config/
COPY utils/ ./utils/

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
