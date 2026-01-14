import pytest
from src.services.confidence_scorer import ConfidenceScorer


@pytest.fixture
def scorer():
    return ConfidenceScorer()


def test_high_confidence(scorer):
    result = scorer.calculate(
        intent_confidence=0.95,
        response_certainty=0.9,
        sentiment=0.5,
        question_count=1,
    )
    assert result.overall >= 0.85
    assert not scorer.should_escalate(result)


def test_low_confidence_escalates(scorer):
    result = scorer.calculate(
        intent_confidence=0.4,
        response_certainty=0.5,
        sentiment=-0.5,
        question_count=3,
    )
    assert result.overall < 0.7
    assert scorer.should_escalate(result)


def test_negative_sentiment_reduces_confidence(scorer):
    positive = scorer.calculate(
        intent_confidence=0.8,
        response_certainty=0.8,
        sentiment=0.5,
        question_count=1,
    )

    negative = scorer.calculate(
        intent_confidence=0.8,
        response_certainty=0.8,
        sentiment=-0.8,
        question_count=1,
    )

    assert negative.overall < positive.overall


def test_multiple_questions_reduce_confidence(scorer):
    single = scorer.calculate(
        intent_confidence=0.8,
        response_certainty=0.8,
        sentiment=0.0,
        question_count=1,
    )

    multiple = scorer.calculate(
        intent_confidence=0.8,
        response_certainty=0.8,
        sentiment=0.0,
        question_count=4,
    )

    assert multiple.overall < single.overall


def test_caveat_threshold(scorer):
    result = scorer.calculate(
        intent_confidence=0.75,
        response_certainty=0.78,
        sentiment=0.2,
        question_count=1,
    )
    assert not scorer.should_escalate(result)
    assert scorer.should_add_caveat(result)
