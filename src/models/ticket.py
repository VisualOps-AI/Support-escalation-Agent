from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class TicketSource(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    SLACK = "slack"
    API = "api"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCreate(BaseModel):
    source: TicketSource = TicketSource.API
    customer_id: str
    subject: str
    body: str
    metadata: dict = Field(default_factory=dict)


class ParsedTicket(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: TicketSource
    customer_id: str
    subject: str
    body: str
    sentiment: float = 0.0
    urgency: Urgency = Urgency.MEDIUM
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    status: TicketStatus = TicketStatus.OPEN
    assigned_to: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


class ConversationMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticket_id: str
    role: str  # customer, agent, human
    content: str
    confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    message: str
    confidence: float
    intent: str
    should_escalate: bool = False
    escalation_reason: Optional[str] = None
    suggested_actions: list[str] = Field(default_factory=list)
