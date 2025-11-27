# Backend (FastAPI)

Vietnamese Sentiment Assistant backend built with FastAPI, Hugging Face Transformers, and SQLite persistence.

## Quick start

```pwsh
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Environment variables:

| Name | Default | Notes |
| --- | --- | --- |
| `SENTIMENT_MODEL_NAME` | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | Any sentiment checkpoint compatible with the pipeline API. |
| `SQLITE_PATH` | `backend/data/sentiment_history.sqlite3` | Automatically created. |
| `MIN_TEXT_LENGTH` | `4` | Reject inputs shorter than this. |
| `HISTORY_SOFT_LIMIT` | `50` | Old rows beyond this number are trimmed. |

## Test quick check

```pwsh
cd backend
pytest
```

## Manual evaluation

Run a lightweight self-check with curated Vietnamese cases:

```pwsh
cd backend
python -m app.tools.evaluate
```
