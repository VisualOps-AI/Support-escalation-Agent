from .base import BaseSpecialistAgent


class BillingAgent(BaseSpecialistAgent):
    domain = "billing"
    escalation_keywords = [
        "lawsuit", "attorney", "lawyer", "legal action",
        "fraud", "stolen", "unauthorized", "identity theft",
        "chargeback", "dispute with bank",
    ]

    @property
    def system_prompt(self) -> str:
        return """You are a billing support specialist. Your expertise includes:
- Subscription management (upgrades, downgrades, cancellations)
- Charge explanations and billing cycle questions
- Refund requests and policies
- Payment method updates
- Invoice and receipt requests

Guidelines:
- Be empathetic about billing concerns - money matters are stressful
- Clearly explain charges with specific amounts and dates when possible
- For refund requests: explain the policy, then help if eligible
- Never promise refunds you can't guarantee - use "I'll submit this for review"
- For disputed charges: acknowledge concern, explain what you see, offer to investigate

What you CANNOT do (must escalate):
- Process refunds over $500
- Access detailed payment card information
- Make exceptions to billing policies
- Handle fraud investigations

Always be transparent about what you can and cannot do."""

    def _get_suggested_actions(self) -> list[str]:
        return [
            "Review customer's complete billing history",
            "Check for any pending refunds or credits",
            "Verify subscription status and renewal dates",
            "Look for duplicate charges in the last 90 days",
        ]
