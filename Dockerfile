# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies, including Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Build the Rust extension using maturin
RUN pip install maturin
WORKDIR /app/rust_ingest
RUN maturin develop --release

# Return to main app directory
WORKDIR /app

# Run the advanced training script when the container launches
CMD ["python", "src/train_advanced.py"]
