FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY exporter.py /app/exporter.py

# nvidia-smi is mounted in by the NVIDIA container toolkit (utility capability)
USER 65534
EXPOSE 9835

CMD ["python3", "/app/exporter.py"]
