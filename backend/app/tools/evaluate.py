"""Utility script to ensure the transformer pipeline keeps the target accuracy."""

from __future__ import annotations

from dataclasses import dataclass

from ..sentiment import SentimentClassifier

TEST_CASES = {
    "Hôm nay tôi rất vui": "POSITIVE",
    "Món ăn này dở quá": "NEGATIVE",
    "Cũng bình thường thôi": "NEUTRAL",
    "Dịch vụ okela, nhân viên dễ thương": "POSITIVE",
    "Chán chẳng muốn nói": "NEGATIVE",
    "Tạm ổn, không có gì đặc biệt": "NEUTRAL",
    "Trải nghiệm thật tuyệt vời": "POSITIVE",
    "Ứng dụng chạy chậm kinh khủng": "NEGATIVE",
    "Khá được, nhưng cần cải thiện": "NEUTRAL",
    "Tôi cực kỳ hài lòng": "POSITIVE",
}


@dataclass
class EvaluationResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_evaluation(model_name: str) -> EvaluationResult:
    classifier = SentimentClassifier(model_name)
    correct = 0
    for text, expected in TEST_CASES.items():
        prediction = classifier.analyze(text)
        if prediction.sentiment == expected:
            correct += 1
        print(f"{text} => {prediction.sentiment} ({prediction.confidence:.2f}) | expected {expected}")
    return EvaluationResult(total=len(TEST_CASES), correct=correct)


if __name__ == "__main__":
    result = run_evaluation("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    accuracy = result.accuracy * 100
    status = "PASS" if accuracy >= 65 else "FAIL"
    print(f"Accuracy: {accuracy:.1f}% ({result.correct}/{result.total}) => {status}")
