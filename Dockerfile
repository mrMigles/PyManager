FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

# git is needed to clone/sync GitHub-based apps
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# python-telegram-bot & psutil are installed system-wide: every per-app venv is
# created with --system-site-packages, so this is the one thing shared across apps.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py pgdb.py ./

VOLUME ["/data"]

CMD ["python", "main.py"]
