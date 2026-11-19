FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY prompts ./prompts
COPY fixtures ./fixtures
RUN pip install . && useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app /app/data

USER app
ENV SDR_ROOT=/app SDR_DB_PATH=/app/data/queue.db
EXPOSE 8000
# Seed the queue with offline demo runs, then serve the approval UI.
CMD ["sh", "-c", "sdr-crew demo --db /app/data/queue.db --trace /app/data/demo.jsonl && sdr-crew serve --host 0.0.0.0 --port 8000 --db /app/data/queue.db"]
