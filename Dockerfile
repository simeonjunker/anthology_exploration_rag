FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PERSIST_DIR=/data/chroma_db \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --locked --no-dev \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data "$HF_HOME" \
    && chown -R app:app /data /home/app

ENV PATH="/app/.venv/bin:$PATH"

USER app

CMD ["chat"]
