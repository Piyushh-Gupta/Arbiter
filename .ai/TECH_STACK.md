# Technology Stack

## Frozen Dependencies
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Database**: SQLite, SQLAlchemy
- **Configuration**: Hydra, Pydantic-Settings
- **Logging**: Structlog
- **Machine Learning**: PyTorch, Transformers, Sentence-Transformers, XGBoost, FAISS, rank-bm25
- **Experiment Tracking**: Weights & Biases (W&B)
- **Tooling**: Poetry, Ruff, Mypy, Pytest

## Frozen Models
- **ASR**: `openai/whisper-base.en`
- **Extractor**: `distilbert-base-uncased` (fine-tuned)
- **Verifier**: `microsoft/deberta-v3-base` (fine-tuned)

*Any addition to this stack requires an explicit architectural review.*
