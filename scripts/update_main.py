#!/usr/bin/env python3
"""Update main.py with Phase 2 content."""

CONTENT = '''from contextlib import asynccontextmanager
from datetime import datetime
import os
from typing import Optional
import anthropic
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

from src.models.database import init_db, get_db, Ticket, Conversation, Escalation
from src.models.ticket import (
    TicketCreate,
    ParsedTicket,
    TicketStatus,
    ConversationMessage,
)
from src.agents.specialists import AgentRouter
from src.knowledge import KnowledgeBase

load_dotenv()

knowledge_base: Optional[KnowledgeBase] = None
router: Optional[AgentRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global knowledge_base, router

    await init_db()

    knowledge_base = KnowledgeBase(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    )

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    router = AgentRouter(client, knowledge_base)

    yield


app = FastAPI(
    title="Support Escalation Agent",
    description="AI-powered customer support with intelligent escalation and specialist routing",
    version="0.2.0",
    lifespan=lifespan,
)


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    assigned_agent: str
    response: str
    confidence: float
    intent: str
    routed_to: str
    escalated: bool


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    response: str
    confidence: float
    routed_to: str
    escalated: bool


class EscalateRequest(BaseModel):
    reason: str


class ResolveRequest(BaseModel):
    resolution: str
    feedback: Optional[str] = None


class AnalyticsSummary(BaseModel):
    total_tickets: int
    open_tickets: int
    resolved_tickets: int
    escalated_tickets: int
    auto_resolved_rate: float
    avg_confidence: float
    tickets_by_domain: dict


class KnowledgeBaseStats(BaseModel):
    collections: list[str]
    document_counts: dict


@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(
    ticket_data: TicketCreate,
    db: AsyncSession = Depends(get_db),
):
    parsed = ParsedTicket(
        source=ticket_data.source,
        customer_id=ticket_data.customer_id,
        subject=ticket_data.subject,
        body=ticket_data.body,
        metadata=ticket_data.metadata,
    )

    response, domain = await router.route(parsed)

    db_ticket = Ticket(
        id=parsed.id,
        source=parsed.source.value,
        customer_id=parsed.customer_id,
        subject=parsed.subject,
        body=parsed.body,
        sentiment=parsed.sentiment,
        urgency=parsed.urgency.value,
        intent=response.intent,
        intent_confidence=response.confidence,
        status=TicketStatus.ESCALATED.value if response.should_escalate else TicketStatus.IN_PROGRESS.value,
        assigned_to=f"{domain}_agent" if not response.should_escalate else None,
        metadata_={"routed_to": domain, **parsed.metadata},
    )
    db.add(db_ticket)

    customer_msg = Conversation(
        id=ConversationMessage(
            ticket_id=parsed.id,
            role="customer",
            content=f"Subject: {parsed.subject}\\n\\n{parsed.body}",
        ).id,
        ticket_id=parsed.id,
        role="customer",
        content=f"Subject: {parsed.subject}\\n\\n{parsed.body}",
    )
    db.add(customer_msg)

    agent_msg = Conversation(
        id=ConversationMessage(
            ticket_id=parsed.id,
            role="agent",
            content=response.message,
            confidence=response.confidence,
        ).id,
        ticket_id=parsed.id,
        role="agent",
        content=response.message,
        confidence=response.confidence,
    )
    db.add(agent_msg)

    if response.should_escalate:
        escalation = Escalation(
            id=parsed.id + "-esc",
            ticket_id=parsed.id,
            reason=response.escalation_reason or "Low confidence",
            context_package={
                "intent": response.intent,
                "confidence": response.confidence,
                "routed_to": domain,
                "suggested_actions": response.suggested_actions,
            },
        )
        db.add(escalation)

    await db.commit()

    return TicketResponse(
        ticket_id=parsed.id,
        status=db_ticket.status,
        assigned_agent=f"{domain}_agent" if not response.should_escalate else "pending_human",
        response=response.message,
        confidence=response.confidence,
        intent=response.intent,
        routed_to=domain,
        escalated=response.should_escalate,
    )


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    conv_result = await db.execute(
        select(Conversation)
        .where(Conversation.ticket_id == ticket_id)
        .order_by(Conversation.created_at)
    )
    conversations = conv_result.scalars().all()

    esc_result = await db.execute(
        select(Escalation).where(Escalation.ticket_id == ticket_id)
    )
    escalation = esc_result.scalar_one_or_none()

    return {
        "ticket": {
            "id": ticket.id,
            "source": ticket.source,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "body": ticket.body,
            "intent": ticket.intent,
            "status": ticket.status,
            "assigned_to": ticket.assigned_to,
            "routed_to": ticket.metadata_.get("routed_to", "unknown"),
            "created_at": ticket.created_at.isoformat(),
        },
        "conversations": [
            {
                "role": c.role,
                "content": c.content,
                "confidence": c.confidence,
                "created_at": c.created_at.isoformat(),
            }
            for c in conversations
        ],
        "escalation": {
            "reason": escalation.reason,
            "suggested_actions": escalation.context_package.get("suggested_actions", []),
            "created_at": escalation.created_at.isoformat(),
        } if escalation else None,
    }


@app.post("/tickets/{ticket_id}/message", response_model=MessageResponse)
async def send_message(
    ticket_id: str,
    message: MessageRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if ticket.status == TicketStatus.ESCALATED.value:
        raise HTTPException(status_code=400, detail="Ticket is escalated to human agent")

    conv_result = await db.execute(
        select(Conversation)
        .where(Conversation.ticket_id == ticket_id)
        .order_by(Conversation.created_at)
    )
    conversations = conv_result.scalars().all()

    history = []
    for conv in conversations:
        role = "user" if conv.role == "customer" else "assistant"
        history.append({"role": role, "content": conv.content})

    parsed = ParsedTicket(
        id=ticket.id,
        source=ticket.source,
        customer_id=ticket.customer_id,
        subject=ticket.subject,
        body=message.content,
        intent=ticket.intent,
        intent_confidence=ticket.intent_confidence,
    )

    domain = ticket.metadata_.get("routed_to", "general")
    agent = router.get_agent_for_domain(domain)
    response = await agent.handle(parsed, history)

    customer_msg = Conversation(
        id=ConversationMessage(ticket_id=ticket_id, role="customer", content=message.content).id,
        ticket_id=ticket_id,
        role="customer",
        content=message.content,
    )
    db.add(customer_msg)

    agent_msg = Conversation(
        id=ConversationMessage(
            ticket_id=ticket_id,
            role="agent",
            content=response.message,
            confidence=response.confidence,
        ).id,
        ticket_id=ticket_id,
        role="agent",
        content=response.message,
        confidence=response.confidence,
    )
    db.add(agent_msg)

    if response.should_escalate:
        ticket.status = TicketStatus.ESCALATED.value
        escalation = Escalation(
            id=ticket_id + f"-esc-{datetime.utcnow().timestamp()}",
            ticket_id=ticket_id,
            reason=response.escalation_reason or "Low confidence during conversation",
            context_package={
                "intent": response.intent,
                "confidence": response.confidence,
                "routed_to": domain,
                "suggested_actions": response.suggested_actions,
                "conversation_length": len(conversations) + 2,
            },
        )
        db.add(escalation)

    await db.commit()

    return MessageResponse(
        response=response.message,
        confidence=response.confidence,
        routed_to=domain,
        escalated=response.should_escalate,
    )


@app.post("/tickets/{ticket_id}/escalate")
async def escalate_ticket(
    ticket_id: str,
    request: EscalateRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = TicketStatus.ESCALATED.value

    escalation = Escalation(
        id=ticket_id + f"-esc-manual-{datetime.utcnow().timestamp()}",
        ticket_id=ticket_id,
        reason=f"Manual escalation: {request.reason}",
        context_package={"manual": True},
    )
    db.add(escalation)
    await db.commit()

    return {"status": "escalated", "ticket_id": ticket_id}


@app.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: str,
    request: ResolveRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = TicketStatus.RESOLVED.value
    ticket.resolved_at = datetime.utcnow()

    resolution_msg = Conversation(
        id=ConversationMessage(
            ticket_id=ticket_id,
            role="human",
            content=f"[RESOLVED] {request.resolution}",
        ).id,
        ticket_id=ticket_id,
        role="human",
        content=f"[RESOLVED] {request.resolution}",
    )
    db.add(resolution_msg)
    await db.commit()

    return {"status": "resolved", "ticket_id": ticket_id}


@app.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics(db: AsyncSession = Depends(get_db)):
    total_result = await db.execute(select(Ticket))
    all_tickets = total_result.scalars().all()

    total = len(all_tickets)
    if total == 0:
        return AnalyticsSummary(
            total_tickets=0,
            open_tickets=0,
            resolved_tickets=0,
            escalated_tickets=0,
            auto_resolved_rate=0.0,
            avg_confidence=0.0,
            tickets_by_domain={},
        )

    open_count = sum(1 for t in all_tickets if t.status == TicketStatus.OPEN.value)
    resolved_count = sum(1 for t in all_tickets if t.status == TicketStatus.RESOLVED.value)
    escalated_count = sum(1 for t in all_tickets if t.status == TicketStatus.ESCALATED.value)

    resolved_without_escalation = sum(
        1 for t in all_tickets
        if t.status == TicketStatus.RESOLVED.value
    )

    auto_resolved_rate = resolved_without_escalation / total if total > 0 else 0.0
    avg_confidence = sum(t.intent_confidence for t in all_tickets) / total

    domain_counts = {}
    for t in all_tickets:
        domain = t.metadata_.get("routed_to", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    return AnalyticsSummary(
        total_tickets=total,
        open_tickets=open_count,
        resolved_tickets=resolved_count,
        escalated_tickets=escalated_count,
        auto_resolved_rate=auto_resolved_rate,
        avg_confidence=avg_confidence,
        tickets_by_domain=domain_counts,
    )


@app.get("/knowledge/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_stats():
    collections = knowledge_base.list_collections()
    counts = {}
    for coll in collections:
        counts[coll] = await knowledge_base.get_collection_count(coll)

    return KnowledgeBaseStats(
        collections=collections,
        document_counts=counts,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.2.0"}
'''

if __name__ == "__main__":
    from pathlib import Path
    target = Path(__file__).parent.parent / "src" / "api" / "main.py"
    target.write_text(CONTENT)
    print(f"Updated {target}")
