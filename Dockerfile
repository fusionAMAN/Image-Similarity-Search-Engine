# Base image
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose internal port (Render maps it dynamically)
EXPOSE 5000

# Start Gunicorn in shell form so $PORT expands
CMD gunicorn --workers 2 --bind 0.0.0.0:$PORT --timeout 120 app:app
