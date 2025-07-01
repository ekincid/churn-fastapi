FROM python:3.11-slim

WORKDIR /app

COPY inference.py .
COPY exported_model/ ./model

# Modelin kendi environment'ı içindeki requirements
COPY exported_model/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
