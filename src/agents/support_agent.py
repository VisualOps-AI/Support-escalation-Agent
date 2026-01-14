import anthropic
from src.models.ticket import ParsedTicket, AgentResponse
from src.services.intent_classifier import IntentClassifier, Intent
from src.services.confidence_scorer import ConfidenceScorer, ConfidenceBreakdown


SYSTEM_PROMPT = """You are a helpful customer support agent. Your role is to assist customers with their inquiries professionally and efficiently.

Guidelines:
- Be concise but thorough
- Show empathy when appropriate
- If you're unsure about something, acknowledge it
- Never make up information about policies, prices, or technical details
- For billing issues: explain clearly, offer to help resolve
- For technical issues: ask clarifying questions if needed, provide step-by-step guidance
- For account issues: prioritize security while being helpful

When you cannot fully resolve an issue, acknowledge the limitation and explain what you can do.

Respond directly to the customer. Do not include any metadata or confidence scores in your response."""


class SupportAgent:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client
        self.classifier = IntentClassifier(client)
        self.scorer = ConfidenceScorer()

    async def handle_ticket(
        self,
        ticket: ParsedTicket,
        conversation_history: list[dict] | None = None,
    ) -> AgentResponse:
        intent = await self.classifier.classify(ticket.subject, ticket.body)
        response_text, response_certainty = await self._generate_response(
            ticket, intent, conversation_history
        )

        question_count = ticket.body.count("?")
        confidence = self.scorer.calculate(
            intent_confidence=intent.confidence,
            response_certainty=response_certainty,
            sentiment=ticket.sentiment,
            question_count=question_count,
        )

        should_escalate = self.scorer.should_escalate(confidence)
        final_response = self._prepare_final_response(
            response_text, confidence, should_escalate
        )

        return AgentResponse(
            message=final_response,
            confidence=confidence.overall,
            intent=intent.category.value,
            should_escalate=should_escalate,
            escalation_reason=self._get_escalation_reason(confidence) if should_escalate else None,
            suggested_actions=self._get_suggested_actions(intent, should_escalate),
        )

    async def _generate_response(
        self,
        ticket: ParsedTicket,
        intent: Intent,
        conversation_history: list[dict] | None,
    ) -> tuple[str, float]:
        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        user_message = f"Subject: {ticket.subject}\n\n{ticket.body}"
        messages.append({"role": "user", "content": user_message})

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        response_text = response.content[0].text
        certainty = self._estimate_response_certainty(response_text, intent)

        return response_text, certainty

    def _estimate_response_certainty(self, response: str, intent: Intent) -> float:
        base_certainty = 0.8

        uncertain_phrases = [
            "i'm not sure",
            "i don't know",
            "i cannot",
            "unable to",
            "might be",
            "could be",
            "possibly",
            "i think",
            "i believe",
        ]

        lower_response = response.lower()
        uncertainty_count = sum(1 for phrase in uncertain_phrases if phrase in lower_response)

        certainty_penalty = uncertainty_count * 0.1
        certainty = max(0.3, base_certainty - certainty_penalty)

        if intent.confidence < 0.5:
            certainty *= 0.8

        return certainty

    def _prepare_final_response(
        self,
        response: str,
        confidence: ConfidenceBreakdown,
        should_escalate: bool,
    ) -> str:
        if should_escalate:
            return (
                f"{response}\n\n"
                "I want to make sure you get the best help possible. "
                "I'm connecting you with a specialist who can better assist with your specific situation."
            )

        if self.scorer.should_add_caveat(confidence):
            return (
                f"{response}\n\n"
                "Please let me know if this doesn't fully address your question, "
                "and I'll be happy to help further or connect you with a specialist."
            )

        return response

    def _get_escalation_reason(self, confidence: ConfidenceBreakdown) -> str:
        reasons = []
        if confidence.intent_clarity < 0.6:
            reasons.append("unclear customer intent")
        if confidence.response_certainty < 0.6:
            reasons.append("low response confidence")
        if confidence.sentiment_risk < 0.5:
            reasons.append("negative customer sentiment")
        if confidence.complexity_factor < 0.6:
            reasons.append("complex multi-part inquiry")
        return "; ".join(reasons) if reasons else "overall low confidence"

    def _get_suggested_actions(self, intent: Intent, escalated: bool) -> list[str]:
        if not escalated:
            return []

        suggestions = {
            "billing": [
                "Review customer's billing history",
                "Check for recent charge disputes",
                "Verify subscription status",
            ],
            "technical": [
                "Check system status page",
                "Review customer's recent activity logs",
                "Verify account configuration",
            ],
            "account": [
                "Verify customer identity",
                "Check account security flags",
                "Review recent login attempts",
            ],
        }

        category_prefix = intent.category.value.split(".")[0]
        return suggestions.get(category_prefix, ["Review full ticket history"])
