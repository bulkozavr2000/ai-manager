# The agent on a Linux server (ruling R86): Python 3.12, the dependencies from uv.lock,
# Playwright's own Chromium with its system libraries, a non-root user.
#
#   docker compose up -d --build
#
# Nothing secret is in the image: the keys are in data/keys.toml (python -m tg_agent.setup
# export-keys on the Windows computer, or setup secrets inside the container), and the data
# folder and config.toml are mounted from the host (compose.yaml).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv

# No key store in a container (PYTHON_KEYRING_BACKEND): the keys come from data/keys.toml.
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# The dependencies first, so a change of the code alone does not install them again.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Chromium and the libraries it needs (apt, as root), into a folder every user can read.
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY src ./src
COPY config.example.toml ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app/data /app/tools \
    && chown agent:agent /app/data
USER agent

# Ctrl+C, not a kill: the agent closes the database and the Telegram link before it ends.
STOPSIGNAL SIGINT
CMD ["python", "-m", "tg_agent"]
