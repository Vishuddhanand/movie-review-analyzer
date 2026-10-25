FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# models/*.pkl and nltk_data/ are copied in, so the container needs no internet at runtime
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/healthz')" || exit 1

CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT}"]
