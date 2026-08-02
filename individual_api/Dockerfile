FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-keirin.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY keirin_individual_api.py .
COPY keirin_pdf_adapter.py .
COPY keirin_jp_pdf_adapter.py .
COPY keirin_jp_6pdf_adapter.py .
COPY keirin_jp_5pdf_adapter.py .
COPY keirin_dual_strategy_api.py .
COPY keirin_dual_pdf_adapter.py .
COPY keirin_api_server.py .

ENV KEIRIN_API_HOST=0.0.0.0
ENV KEIRIN_API_PORT=8787

EXPOSE 8787
CMD ["python", "keirin_api_server.py"]
