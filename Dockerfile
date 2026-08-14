FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Install system dependencies required for spatial index (OSMnx) and standard builds
RUN apt-get update && apt-get install -y \
    build-essential \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies (no-cache-dir keeps the image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
