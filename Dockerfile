FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train.py .
COPY prometheus.yml .

EXPOSE 8000

CMD ["python", "app.py"]
