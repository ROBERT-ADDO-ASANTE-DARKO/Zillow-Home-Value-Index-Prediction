FROM python:3.11.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render exposes PORT env var; default to 8000
ENV PORT=8000

EXPOSE ${PORT}

CMD gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --log-level info
