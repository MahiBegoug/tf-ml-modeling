# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g. for lightgbm sometimes)
# curl, unzip required for rclone
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Rclone and gdown (for easier drive downloads)
RUN curl https://rclone.org/install.sh | bash \
    && pip install gdown

# Copy configuration files first for better caching
COPY pyproject.toml setup.py requirements.txt ./

# Install dependencies (including the package itself in editable mode if desired, 
# or standard install. Here we do a standard install of the current directory)
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Set entrypoint so arguments are appended
ENTRYPOINT ["python", "run_ml.py"]
CMD ["--help"]
