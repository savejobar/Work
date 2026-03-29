FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=200 --no-cache-dir \
    -r requirements.txt
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]