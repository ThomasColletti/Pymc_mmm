# Use slim Python image
FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + model artifacts
COPY server.py .
COPY artifacts ./artifacts

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
