# Vietnamese Sentiment Assistant

Full-stack project that exposes a FastAPI backend powered by Hugging Face Transformers and a React frontend for classifying Vietnamese sentences into POSITIVE/NEUTRAL/NEGATIVE labels. The backend stores every classification inside a local SQLite database so users can review previous analyses.

## Repository structure

```
backend/   # FastAPI service, Hugging Face pipeline, SQLite persistence
frontend/  # React UI (Vite) with history table and pop-up validation errors
```

## Getting started

1. **Backend** – see `backend/README.md` for environment setup. Start the API with `uvicorn app.main:app --reload --port 8000`.
2. **Frontend** – see `frontend/README.md` (created via Vite). Run `npm install` then `npm run dev`.

Default API base URL: `http://localhost:8000`. The frontend dev server expects this value and proxies requests under `/api`.

## Accuracy target

Use `python -m app.tools.evaluate` under `backend/` to run 10 curated Vietnamese sentences. The script prints coverage and ensures accuracy stays above 65%.
