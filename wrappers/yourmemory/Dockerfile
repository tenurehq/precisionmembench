FROM python:3.13-slim
WORKDIR /app
RUN pip install fastapi==0.136.3 uvicorn==0.47.0 httpx==0.28.1
COPY yourmemory_service.py .
CMD ["python", "yourmemory_service.py"]