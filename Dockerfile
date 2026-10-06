# NOTE: written carefully but NOT build-tested in the environment this was
# authored in (no Docker available there) - run `docker build .` yourself
# as the first step before relying on this. See README's "Serving" for
# the full deployment story.

FROM python:3.11-slim

# build-essential: needed for some wheels (e.g. sentencepiece) that may
# not have a prebuilt wheel for every platform. Removed in the same layer
# it's installed in isn't possible with apt's own cache cleanup alone, but
# the base image is `slim` and this is the only extra system dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-serving.txt

COPY pyproject.toml ./
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY evaluation/ evaluation/

# Pinned to bge-base rather than the Python-level default (bge-large,
# ~1.3GB) - this image targets Render's `starter` plan (see render.yaml),
# and bge-large's memory footprint alongside PyTorch/FastAPI overhead isn't
# verified to fit there. bge-base (~420MB) is the safer default for this
# specific deployment target; a beefier instance can override EMBEDDING_MODEL
# at build time (see README's "Deploying" for the full tradeoff). This must
# be set BEFORE the prefetch/build-index steps below so the model cached
# into the image and the one the running container expects always match -
# retrieval.load_index() refuses to load an index built with a different
# model than the one currently configured.
ENV EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# The configured embedding model is needed for every query embedding.
# Cache it in /app (owned by appuser below) during the image build so the
# first visitor cannot trigger an unexpected Hugging Face download. The
# Gemma model remains external: it is gated and far too large to bake into
# this public image.
ENV HF_HOME=/app/.cache/huggingface
RUN PYTHONPATH=src python -c "from chatbot_rag.retrieval import _get_embedding_model; _get_embedding_model()"

# No index is baked into the image: facts and embeddings live in the
# database (DATABASE_URL, injected at runtime). Updating the knowledge base is
# `python scripts/sync_facts.py`, not a rebuild/redeploy.

# Render uses LLM_BACKEND=api, so no generation model weights are baked into
# this image. The Gemini API key is injected at runtime as a secret.


# Hugging Face Spaces runs the container as UID 1000, so create the user with
# that exact UID (also fine on any other Docker host).
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces expects the app on port 7860 (README.md: app_port).
# Other hosts (e.g. Render) inject their own PORT at runtime, which wins.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'), timeout=3)" || exit 1

# Listen on all interfaces so the host's proxy can reach the container.
ENV HOST=0.0.0.0

CMD ["python", "scripts/serve.py"]
