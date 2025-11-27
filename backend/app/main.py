"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import HistoryRepository
from .schemas import HistoryEntry as HistorySchema
from .schemas import SentimentRequest, SentimentResponse
from .sentiment import SentimentClassifier

LOGGER = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self.repo = HistoryRepository(settings.sqlite_path, settings.history_soft_limit)
        self.classifier = SentimentClassifier(settings.model_name)


def create_app() -> FastAPI:
    app = FastAPI(title="Vietnamese Sentiment Assistant", version="0.1.0")
    state = AppState()

    app.state.repo = state.repo
    app.state.classifier = state.classifier

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    def get_repo(request: Request) -> HistoryRepository:
        return request.app.state.repo

    def get_classifier(request: Request) -> SentimentClassifier:
        return request.app.state.classifier

    @app.post("/api/sentiment", response_model=SentimentResponse)
    async def analyze_sentiment(
        payload: SentimentRequest,
        repo: HistoryRepository = Depends(get_repo),
        classifier: SentimentClassifier = Depends(get_classifier),
    ) -> SentimentResponse:
        cleaned = payload.text.strip()
        if len(cleaned) < settings.min_text_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Câu quá ngắn! Hãy nhập tối thiểu %d ký tự." % settings.min_text_length,
            )
        result = classifier.analyze(cleaned)
        entry = repo.add_entry(result.text, result.sentiment, result.confidence)
        return SentimentResponse(
            text=entry.text,
            sentiment=entry.sentiment,  # type: ignore[arg-type]
            confidence=entry.confidence,
            created_at=entry.created_at,
        )

    @app.get("/api/history", response_model=list[HistorySchema])
    async def read_history(
        limit: int = Query(20, ge=1, le=200),
        repo: HistoryRepository = Depends(get_repo),
    ) -> list[HistorySchema]:
        entries = repo.list_entries(limit)
        return [
            HistorySchema(
                id=entry.id,
                text=entry.text,
                sentiment=entry.sentiment,  # type: ignore[arg-type]
                confidence=entry.confidence,
                created_at=entry.created_at,
            )
            for entry in entries
        ]

    return app


app = create_app()
