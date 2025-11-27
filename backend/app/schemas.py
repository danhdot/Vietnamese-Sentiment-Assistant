"""Pydantic schemas for FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Vietnamese sentence to analyze")


class SentimentResponse(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime


class HistoryEntry(SentimentResponse):
    id: int
