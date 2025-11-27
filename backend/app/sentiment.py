"""Sentiment classification helpers built on top of Hugging Face pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from transformers import AutoTokenizer, pipeline

from . import text_utils
from .schemas import SentimentLabel

LOGGER = logging.getLogger(__name__)

# Accented and non-accented variants are accent-folded before lookup.
POSITIVE_LEXICON = {
    "vui",
    "vuiqua",
    "tuyet",
    "tuyetvoi",
    "tot",
    "rattot",
    "hao",
    "hungthu",
    "khen",
    "ngon",
    "thich",
    "yeu",
    "hailong",
    "cuctot",
    "on",
    "onap",
    "dep",
    "tuyetvoi",
}

NEGATIVE_LEXICON = {
    "ghet",
    "te",
    "tequa",
    "tethat",
    "toite",
    "buon",
    "chan",
    "chanqua",
    "tuc",
    "gian",
    "thatvong",
    "khochiu",
    "kinhkhung",
    "xau",
    "xauxi",
    "do",
    "dote",
    "tehai",
}

LABEL_ALIASES: dict[str, SentimentLabel] = {
    "positive": "POSITIVE",
    "positive sentiment": "POSITIVE",
    "neg": "NEGATIVE",
    "negative": "NEGATIVE",
    "negative sentiment": "NEGATIVE",
    "neutral": "NEUTRAL",
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
    "LABEL_2": "NEUTRAL",
    "1 star": "NEGATIVE",
    "2 stars": "NEGATIVE",
    "3 stars": "NEUTRAL",
    "4 stars": "POSITIVE",
    "5 stars": "POSITIVE",
}


@dataclass(slots=True)
class SentimentResult:
    text: str
    sentiment: SentimentLabel
    confidence: float


class SentimentClassifier:
    """Wrapper around a Hugging Face pipeline with light calibration."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = self._load_pipeline(model_name)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_pipeline(model_name: str):
        LOGGER.info("Loading sentiment pipeline with model %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)

    def analyze(self, text: str) -> SentimentResult:
        cleaned = text_utils.normalize_text(text)
        raw = self._pipeline(cleaned[:512])[0]
        base_label = self._normalize_label(str(raw.get("label", "neutral")))
        confidence = float(raw.get("score", 0.0))
        lexical_score = self._lexical_score(cleaned)
        sentiment = self._calibrated_label(base_label, confidence, lexical_score)
        calibrated_confidence = self._mix_confidence(confidence, lexical_score)
        return SentimentResult(text=text.strip(), sentiment=sentiment, confidence=calibrated_confidence)

    def _normalize_label(self, label: str) -> SentimentLabel:
        normalized = LABEL_ALIASES.get(label.lower())
        if normalized:
            return normalized
        LOGGER.debug("Falling back to NEUTRAL for unknown label: %s", label)
        return "NEUTRAL"

    def _lexical_score(self, text: str) -> float:
        folded = text_utils.strip_accents(text)
        tokens = folded.split()
        if not tokens:
            return 0.0
        score = 0
        for token in tokens:
            if token in POSITIVE_LEXICON:
                score += 1
            elif token in NEGATIVE_LEXICON:
                score -= 1
        return score / max(len(tokens), 1)

    def _calibrated_label(
        self, base_label: SentimentLabel, confidence: float, lexical_score: float
    ) -> SentimentLabel:
        if lexical_score >= 0.3:
            return "POSITIVE"
        if lexical_score <= -0.3:
            return "NEGATIVE"
        if base_label == "NEUTRAL" or confidence < 0.55:
            if abs(lexical_score) < 0.15:
                return "NEUTRAL"
        return base_label

    def _mix_confidence(self, confidence: float, lexical_score: float) -> float:
        lexical_boost = min(abs(lexical_score) * 4, 0.35)
        return max(0.0, min(1.0, confidence * 0.7 + lexical_boost))
