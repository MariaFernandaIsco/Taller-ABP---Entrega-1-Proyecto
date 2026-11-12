FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
RUN useradd -m app
WORKDIR /srv
COPY requirements.txt .
RUN grep -vE "^(#|jupyter|pytest|httpx|matplotlib|pandas)" requirements.txt > req-run.txt && pip install --no-cache-dir -r req-run.txt
COPY src ./src
COPY app ./app
COPY data ./data
COPY results/modelo_final.keras results/vocabulario.json ./results/
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
