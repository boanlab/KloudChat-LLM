# Playwright base: Chromium and its system dependencies
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Browser check and config generation
RUN crawl4ai-setup || true

# Adult sites a scrape refuses, from a pinned commit of StevenBlack/hosts (porn-only)
ADD https://raw.githubusercontent.com/StevenBlack/hosts/83dd698bbcdcb8c11a7796af7188d7ab4ccd02f1/alternates/porn-only/hosts /app/adult-hosts.txt

COPY app.py netguard.py adultlist.py gate.py /app/

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
