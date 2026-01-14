from abc import ABC, abstractmethod
from dataclasses import dataclass
import anthropic

from src.models.ticket import ParsedTicket, AgentResponse
from src.services.confidence_scorer import ConfidenceScorer


@dataclass
class RetrievedContext:
    content: str
    source: str
    relevance_score: float


class BaseSpecialistAgent(ABC):
    domain: str
    escalation_keywords: list[str] = []
    max_turns: int = 3

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        knowledge_base=None,
    ):
        self.client = client
        self.knowledge_base = knowledge_base
        self.scorer = ConfidenceScorer()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    def domain_knowledge_collection(self) -> str:
        return f"{self.domain}_knowledge"

    async def handle(
        self,
        ticket: ParsedTicket,
        conversation_history: list[dict] | None = None,
    ) -> AgentResponse:
        retrieved_context = await self._retrieve_context(ticket)

        should_force_escalate = self._check_escalation_triggers(ticket)
        if should_force_escalate:
            return self._create_escalation_response(
                "Detected keywords requiring human specialist attention"
            )

        response_text, certainty = await self._generate_response(
            ticket, retrieved_context, conversation_history
        )

        confidence = self.scorer.calculate(
            intent_confidence=ticket.intent_confidence,
            response_certainty=certainty,
            sentiment=ticket.sentiment,
            question_count=ticket.body.count("?"),
        )

        should_escalate = self.scorer.should_escalate(confidence)
        final_response = self._prepare_response(response_text, confidence, should_escalate)

        return AgentResponse(
            message=final_response,
            confidence=confidence.overall,
            intent=ticket.intent or "unknown",
            should_escalate=should_escalate,
            escalation_reason=self._get_escalation_reason(confidence) if should_escalate else None,
            suggested_actions=self._get_suggested_actions() if should_escalate else [],
        )

    async def _retrieve_context(self, ticket: ParsedTicket) -> list[RetrievedContext]:
        if not self.knowledge_base:
            return []

        query = f"{ticket.subject} {ticket.body}"
        results = await self.knowledge_base.search(
            collection=self.domain_knowledge_collection,
            query=query,
            top_k=3,
        )

        return [
            RetrievedContext(
                content=r["content"],
                source=r.get("source", "knowledge_base"),
                relevance_score=r["score"],
            )
            for r in results
        ]

    async def _generate_response(
        self,
        ticket: ParsedTicket,
        context: list[RetrievedContext],
        conversation_history: list[dict] | None,
    ) -> tuple[str, float]:
        system = self._build_system_prompt(context)

        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        user_message = f"Subject: {ticket.subject}\n\n{ticket.body}"
        messages.append({"role": "user", "content": user_message})

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system,
            messages=messages,
        )

        response_text = response.content[0].text
        certainty = self._estimate_certainty(response_text, context)

        return response_text, certainty

    def _build_system_prompt(self, context: list[RetrievedContext]) -> str:
        base = self.system_prompt

        if context:
            context_block = "\n\n---\nRELEVANT KNOWLEDGE BASE INFORMATION:\n"
            for ctx in context:
                context_block += f"\n[Source: {ctx.source}]\n{ctx.content}\n"
            context_block += "---\n\nUse the above information to inform your response when relevant."
            return base + context_block

        return base

    def _estimate_certainty(
        self,
        response: str,
        context: list[RetrievedContext],
    ) -> float:
        base_certainty = 0.8

        if context:
            avg_relevance = sum(c.relevance_score for c in context) / len(context)
            if avg_relevance > 0.8:
                base_certainty += 0.1
            elif avg_relevance < 0.5:
                base_certainty -= 0.1

        uncertain_phrases = [
            "i'm not sure", "i don't know", "i cannot", "unable to",
            "might be", "could be", "possibly", "i think", "i believe",
        ]
        lower_response = response.lower()
        uncertainty_count = sum(1 for p in uncertain_phrases if p in lower_response)
        base_certainty -= uncertainty_count * 0.1

        return max(0.3, min(1.0, base_certainty))

    def _check_escalation_triggers(self, ticket: ParsedTicket) -> bool:
        text = f"{ticket.subject} {ticket.body}".lower()
        return any(keyword.lower() in text for keyword in self.escalation_keywords)

    def _prepare_response(self, response: str, confidence, should_escalate: bool) -> str:
        if should_escalate:
            return (
                f"{response}\n\n"
                "I want to ensure you get the best possible help. "
                "I'm connecting you with a specialist who can better assist with your specific situation."
            )

        if self.scorer.should_add_caveat(confidence):
            return (
                f"{response}\n\n"
                "Please let me know if this doesn't fully address your question."
            )

        return response

    def _create_escalation_response(self, reason: str) -> AgentResponse:
        return AgentResponse(
            message="I'll connect you with a specialist who can best help with this matter.",
            confidence=0.0,
            intent="escalated",
            should_escalate=True,
            escalation_reason=reason,
            suggested_actions=self._get_suggested_actions(),
        )

    def _get_escalation_reason(self, confidence) -> str:
        reasons = []
        if confidence.intent_clarity < 0.6:
            reasons.append("unclear request")
        if confidence.response_certainty < 0.6:
            reasons.append("low response confidence")
        if confidence.sentiment_risk < 0.5:
            reasons.append("customer frustration detected")
        return "; ".join(reasons) if reasons else "overall low confidence"

    @abstractmethod
    def _get_suggested_actions(self) -> list[str]:
        pass
