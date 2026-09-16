# Running the Emotion Intelligence Platform with Docker

Docker gives the project a clean, repeatable environment, so the Streamlit app can be
run without manually installing Python packages on the host machine.

## 1. Build the image

From the project root:

```bash
docker build -t emotion-intelligence-platform .
```

## 2. Start the application

```bash
docker run --rm -p 8501:8501 emotion-intelligence-platform
```

Then open:

**http://localhost:8501**

## 3. Stop the application

Press `Ctrl + C` in the terminal.

## Notes

The Docker image expects the trained model files to be available in the repository's
`models/` directory because the Streamlit application loads models from there.

For example:

```text
models/
├── embedding_model.h5
├── lstm_model.h5
├── gru_model.h5        # optional
└── vocabulary.pkl      # needed when using TextVectorization models
```

The application can still start when an optional model is missing; the model loader
will report which models are available.

## Why Docker?

The main benefit here is reproducibility. Instead of asking every team member to
install the same Python version, TensorFlow version, and Streamlit dependencies
manually, the project can be built from the same `Dockerfile`.

