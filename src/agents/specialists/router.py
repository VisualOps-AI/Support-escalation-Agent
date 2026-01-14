import anthropic

from src.models.ticket import ParsedTicket, AgentResponse
from src.services.intent_classifier import IntentClassifier, IntentCategory
from .base import BaseSpecialistAgent
from .billing_agent import BillingAgent
from .technical_agent import TechnicalAgent
from .account_agent import AccountAgent


DOMAIN_MAPPING = {
    IntentCategory.BILLING_CHARGE_DISPUTE: "billing",
    IntentCategory.BILLING_REFUND_REQUEST: "billing",
    IntentCategory.BILLING_SUBSCRIPTION: "billing",
    IntentCategory.TECHNICAL_BUG_REPORT: "technical",
    IntentCategory.TECHNICAL_HOW_TO: "technical",
    IntentCategory.TECHNICAL_FEATURE_REQUEST: "technical",
    IntentCategory.ACCOUNT_ACCESS_ISSUE: "account",
    IntentCategory.ACCOUNT_DELETION: "account",
    IntentCategory.ACCOUNT_UPDATE: "account",
    IntentCategory.GENERAL_FEEDBACK: "general",
    IntentCategory.GENERAL_OTHER: "general",
    IntentCategory.UNKNOWN: "general",
}


class AgentRouter:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        knowledge_base=None,
    ):
        self.client = client
        self.classifier = IntentClassifier(client)
        self.knowledge_base = knowledge_base

        self.specialists: dict[str, BaseSpecialistAgent] = {
            "billing": BillingAgent(client, knowledge_base),
            "technical": TechnicalAgent(client, knowledge_base),
            "account": AccountAgent(client, knowledge_base),
        }

        self.generalist = self._create_generalist()

    def _create_generalist(self) -> BaseSpecialistAgent:
        from .billing_agent import BillingAgent

        class GeneralistAgent(BillingAgent):
            domain = "general"
            escalation_keywords = []

            @property
            def system_prompt(self) -> str:
                return """You are a general customer support agent. You handle inquiries that don't fit specific categories like billing, technical support, or account management.

Your role:
- Acknowledge and thank customers for feedback
- Answer general questions about the company/product
- Route complex issues to appropriate specialists
- Provide helpful information when possible

Guidelines:
- Be friendly and professional
- If a question seems like it needs a specialist, suggest connecting them
- For feedback: thank them sincerely and explain how it's used
- Don't make promises about features or timelines"""

            def _get_suggested_actions(self) -> list[str]:
                return ["Review ticket for potential specialist routing"]

        return GeneralistAgent(self.client, self.knowledge_base)

    async def route(
        self,
        ticket: ParsedTicket,
        conversation_history: list[dict] | None = None,
    ) -> tuple[AgentResponse, str]:
        intent = await self.classifier.classify(ticket.subject, ticket.body)

        ticket.intent = intent.category.value
        ticket.intent_confidence = intent.confidence

        domain = DOMAIN_MAPPING.get(intent.category, "general")
        agent = self.specialists.get(domain, self.generalist)

        response = await agent.handle(ticket, conversation_history)

        return response, domain

    def get_agent_for_domain(self, domain: str) -> BaseSpecialistAgent:
        return self.specialists.get(domain, self.generalist)
