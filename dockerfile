FROM astral/uv:python3.12-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

COPY ./src ./src
COPY ./static ./static
COPY main.py .

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]