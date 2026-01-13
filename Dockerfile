FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

COPY main.py .

RUN python -m pip install --no-cache-dir python-telegram-bot==21.6 psutil

VOLUME ["/data"]

CMD ["python", "main.py"]
