FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    httpx==0.28.1 \
    pydantic==2.11.4

COPY supermemory_service.py .

ENV SUPERMEMORY_API_KEY=""
ENV SEED_DELAY_MS=0
ENV POLL_TIMEOUT_S=30
ENV PORT=8080

EXPOSE 8080

CMD ["python", "supermemory_service.py"]
