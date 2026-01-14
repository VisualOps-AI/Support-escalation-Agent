from dataclasses import dataclass
from enum import Enum
import anthropic


class IntentCategory(str, Enum):
    BILLING_CHARGE_DISPUTE = "billing.charge_dispute"
    BILLING_REFUND_REQUEST = "billing.refund_request"
    BILLING_SUBSCRIPTION = "billing.subscription"
    TECHNICAL_BUG_REPORT = "technical.bug_report"
    TECHNICAL_HOW_TO = "technical.how_to"
    TECHNICAL_FEATURE_REQUEST = "technical.feature_request"
    ACCOUNT_ACCESS_ISSUE = "account.access_issue"
    ACCOUNT_DELETION = "account.deletion"
    ACCOUNT_UPDATE = "account.update"
    GENERAL_FEEDBACK = "general.feedback"
    GENERAL_OTHER = "general.other"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    category: IntentCategory
    confidence: float
    reasoning: str


INTENT_CATEGORIES_DESC = """
- billing.charge_dispute: Customer disputes a charge, was charged incorrectly, or sees unexpected charges
- billing.refund_request: Customer wants money back or a refund
- billing.subscription: Questions about subscription plans, upgrades, downgrades, cancellation
- technical.bug_report: Something is broken, not working, crashing, or behaving unexpectedly
- technical.how_to: Customer asking how to do something, needs instructions or guidance
- technical.feature_request: Customer wants a new feature or capability
- account.access_issue: Can't log in, password problems, locked out, 2FA issues
- account.deletion: Customer wants to delete their account or data
- account.update: Customer wants to update profile, email, or account details
- general.feedback: Praise, complaints, or general feedback not fitting other categories
- general.other: Doesn't fit any specific category but is a valid support request
- unknown: Intent is completely unclear or message is nonsensical
"""


class IntentClassifier:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def classify(self, subject: str, body: str) -> Intent:
        prompt = f"""Classify this customer support ticket into exactly one category.

TICKET:
Subject: {subject}
Body: {body}

CATEGORIES:
{INTENT_CATEGORIES_DESC}

Respond in this exact format:
CATEGORY: <category_name>
CONFIDENCE: <0.0-1.0>
REASONING: <one sentence explanation>

Be conservative with confidence scores:
- 0.9-1.0: Extremely clear intent
- 0.7-0.9: Fairly clear intent
- 0.5-0.7: Somewhat ambiguous
- Below 0.5: Very unclear"""

        response = await self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response.content[0].text)

    def _parse_response(self, text: str) -> Intent:
        lines = text.strip().split("\n")
        category = IntentCategory.UNKNOWN
        confidence = 0.5
        reasoning = ""

        for line in lines:
            if line.startswith("CATEGORY:"):
                cat_str = line.replace("CATEGORY:", "").strip()
                try:
                    category = IntentCategory(cat_str)
                except ValueError:
                    category = IntentCategory.UNKNOWN
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return Intent(category=category, confidence=confidence, reasoning=reasoning)
