from app.sentiment import SentimentClassifier, SentimentResult


class DummyPipeline:
    def __call__(self, text):
        if "tuyet" in text:
            return [{"label": "positive", "score": 0.9}]
        if "te" in text:
            return [{"label": "negative", "score": 0.85}]
        return [{"label": "neutral", "score": 0.55}]


def test_classifier_combines_pipeline_and_lexicon(monkeypatch):
    monkeypatch.setattr(SentimentClassifier, "_load_pipeline", lambda *_: DummyPipeline())
    classifier = SentimentClassifier(model_name="dummy")
    result: SentimentResult = classifier.analyze("Dịch vụ tuyệt vời")
    assert result.sentiment == "POSITIVE"


def test_classifier_handles_short_neutral(monkeypatch):
    monkeypatch.setattr(SentimentClassifier, "_load_pipeline", lambda *_: DummyPipeline())
    classifier = SentimentClassifier(model_name="dummy")
    result = classifier.analyze("bình thường")
    assert result.sentiment == "NEUTRAL"
