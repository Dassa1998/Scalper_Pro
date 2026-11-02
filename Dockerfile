FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port (Northflank/Render will override this with $PORT)
ENV PORT=8000
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
