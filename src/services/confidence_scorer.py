from dataclasses import dataclass


@dataclass
class ConfidenceBreakdown:
    intent_clarity: float
    response_certainty: float
    sentiment_risk: float
    complexity_factor: float
    overall: float


class ConfidenceScorer:
    WEIGHTS = {
        "intent_clarity": 0.25,
        "response_certainty": 0.35,
        "sentiment_risk": 0.20,
        "complexity_factor": 0.20,
    }

    THRESHOLDS = {
        "auto_respond": 0.85,
        "respond_with_caveat": 0.70,
        "escalate": 0.70,
    }

    def calculate(
        self,
        intent_confidence: float,
        response_certainty: float,
        sentiment: float,
        question_count: int = 1,
    ) -> ConfidenceBreakdown:
        intent_clarity = intent_confidence
        sentiment_risk = self._calculate_sentiment_risk(sentiment)
        complexity_factor = self._calculate_complexity(question_count)

        overall = (
            self.WEIGHTS["intent_clarity"] * intent_clarity
            + self.WEIGHTS["response_certainty"] * response_certainty
            + self.WEIGHTS["sentiment_risk"] * sentiment_risk
            + self.WEIGHTS["complexity_factor"] * complexity_factor
        )

        return ConfidenceBreakdown(
            intent_clarity=intent_clarity,
            response_certainty=response_certainty,
            sentiment_risk=sentiment_risk,
            complexity_factor=complexity_factor,
            overall=overall,
        )

    def _calculate_sentiment_risk(self, sentiment: float) -> float:
        if sentiment >= 0:
            return 1.0
        return max(0.3, 1.0 + sentiment)

    def _calculate_complexity(self, question_count: int) -> float:
        if question_count <= 1:
            return 1.0
        elif question_count == 2:
            return 0.85
        elif question_count == 3:
            return 0.7
        else:
            return 0.5

    def should_escalate(self, breakdown: ConfidenceBreakdown) -> bool:
        return breakdown.overall < self.THRESHOLDS["escalate"]

    def should_add_caveat(self, breakdown: ConfidenceBreakdown) -> bool:
        return (
            breakdown.overall >= self.THRESHOLDS["respond_with_caveat"]
            and breakdown.overall < self.THRESHOLDS["auto_respond"]
        )
