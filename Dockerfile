FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    chromadb \
    pydantic-settings \
    httpx \
    openai \
    anthropic

# Copy app
COPY app/ ./app/
COPY static/ ./static/
COPY chroma_data/ ./chroma_data/

# Expose port
EXPOSE 8765

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
