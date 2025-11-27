# Frontend (React + Vite)

Simple React client for the Vietnamese Sentiment Assistant.

## Commands

```pwsh
cd frontend
npm install
npm run dev    # http://localhost:5173
npm run build  # production build into dist/
```

The dev server proxies `http://localhost:8000/api` via `vite.config.ts`. Adjust the proxy or create an `.env` file if the backend runs on another host/port.
