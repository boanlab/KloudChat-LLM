FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# libgomp: ONNX Runtime's CPU threads
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app.py layout.py /app/

# The models are fetched here, once, and checked against RapidOCR's published
# hashes; the running service never reaches the network for them.
RUN python -c "import app; app.warm()" \
    && useradd --uid 1000 --create-home ocr
USER ocr

EXPOSE 8080

# No access log: the query string names pages of users' documents
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
