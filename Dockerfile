FROM python:3.12-slim

LABEL maintainer="Michael Banf <michael@perelyn.com>"
LABEL description="Causal bQTL Atlas — pipeline + Pydantic AI agent"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Nextflow
RUN curl -s https://get.nextflow.io | bash \
    && mv nextflow /usr/local/bin/

WORKDIR /app

# Copy project definition and lockfile first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --extra agent --no-dev --frozen

# Copy pipeline code (not data — mount as volume)
COPY scripts/ scripts/
COPY atlas_agent.py atlas_tui.py main.nf ./
COPY paper/ paper/

# Create output directories
RUN mkdir -p results figures/paper_figures data/raw data/processed

ENTRYPOINT ["uv", "run"]
CMD ["atlas-agent", "--validate"]
