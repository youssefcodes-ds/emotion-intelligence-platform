## Running with Docker

### Option A — plain Docker

```bash
docker build -t emotion-platform .
docker run -p 8501:8501 emotion-platform
```

Open http://localhost:8501

### Option B — Docker Compose (one command)

```bash
docker compose up --build
```

Stop with `docker compose down`.

### Requirements

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- No local Python install needed — everything runs inside the container

### Notes

- The image installs `requirements.txt` as a separate layer before copying
  application code, so rebuilding after a code-only change is fast.
- `models/` is expected to already contain the trained model files
  (`embedding_model.h5`, `lstm_model.h5`, `vocabulary.pkl`, the classical
  `.joblib` files) before building — these are tracked in git, not
  downloaded at build time.
- `.dockerignore` excludes `data/raw/`, notebooks, and the local `.venv`
  from the image, since none of those are needed to run the app.
