# Customer Support Escalation Agent
## Technical Design Document

---

## 1. Project Overview

**What it does:** An AI agent system that handles tier-1 customer support, intelligently routes complex issues to specialists, and escalates to humans with full context when confidence is low.

**Why it matters:** Companies waste 60-70% of support costs on tickets AI could handle. But naive chatbots frustrate customers. This system knows its limits.

---

## 2. Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Know your limits** | Confidence scoring on every response |
| **Fail gracefully** | Escalate with context, never dead-end |
| **Learn continuously** | Feedback loop from human resolutions |
| **Full transparency** | Audit trail for every decision |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Email   │  │  Chat    │  │  Slack   │  │   API    │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        └─────────────┴──────┬──────┴─────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                        │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Ticket Parser  │───▶│  Intent Router  │                    │
│  │  - Normalize    │    │  - Classify     │                    │
│  │  - Extract meta │    │  - Route        │                    │
│  └─────────────────┘    └────────┬────────┘                    │
│                                  │                              │
│         ┌────────────────────────┼────────────────────────┐    │
│         ▼                        ▼                        ▼    │
│  ┌─────────────┐         ┌─────────────┐         ┌───────────┐ │
│  │   Billing   │         │  Technical  │         │  Account  │ │
│  │    Agent    │         │    Agent    │         │   Agent   │ │
│  └──────┬──────┘         └──────┬──────┘         └─────┬─────┘ │
│         └────────────────────────┼─────────────────────┘       │
│                                  ▼                              │
│                      ┌─────────────────────┐                   │
│                      │  Confidence Scorer  │                   │
│                      │  - Response quality │                   │
│                      │  - Uncertainty      │                   │
│                      └──────────┬──────────┘                   │
│                                 │                               │
│                    ┌────────────┴────────────┐                 │
│                    ▼                         ▼                 │
│             [confidence ≥ 0.8]        [confidence < 0.8]       │
│                    │                         │                 │
│                    ▼                         ▼                 │
│           ┌──────────────┐         ┌─────────────────┐        │
│           │   Respond    │         │ Escalation Mgr  │        │
│           └──────────────┘         │ - Build context │        │
│                                    │ - Select human  │        │
│                                    │ - Queue ticket  │        │
│                                    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LEARNING LAYER                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Feedback Loop   │  │ Resolution DB   │  │ Analytics      │  │
│  │ - Human edits   │  │ - Past tickets  │  │ - Metrics      │  │
│  │ - Corrections   │  │ - Embeddings    │  │ - Trends       │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Specifications

### 4.1 Ticket Parser
**Purpose:** Normalize incoming tickets into standard format

```python
class ParsedTicket:
    id: str
    source: str  # email, chat, slack, api
    customer_id: str
    subject: str
    body: str
    sentiment: float  # -1 to 1
    urgency: str  # low, medium, high, critical
    attachments: list[Attachment]
    metadata: dict  # source-specific data
    created_at: datetime
```

### 4.2 Intent Router
**Purpose:** Classify ticket and route to specialist

**Categories:**

| Intent | Route To | Examples |
|--------|----------|----------|
| `billing.charge_dispute` | Billing Agent | "I was charged twice" |
| `billing.refund_request` | Billing Agent | "I want my money back" |
| `technical.bug_report` | Technical Agent | "App crashes when I..." |
| `technical.how_to` | Technical Agent | "How do I export data?" |
| `account.access_issue` | Account Agent | "Can't log in" |
| `account.deletion` | Account Agent | "Delete my account" |
| `general.feedback` | Auto-respond | "Love the product!" |
| `unknown` | Escalate | Unclear intent |

**Implementation:** Fine-tuned classifier or LLM with structured output

### 4.3 Specialist Agents
**Purpose:** Domain-specific response generation

Each agent has:
- **System prompt** with domain knowledge
- **Tool access** (knowledge base search, account lookup, etc.)
- **Response templates** for common scenarios
- **Escalation triggers** (keywords, complexity signals)

```python
class SpecialistAgent:
    domain: str
    system_prompt: str
    tools: list[Tool]
    escalation_keywords: list[str]
    max_turns: int = 3  # before forced escalation

    async def handle(
        self,
        ticket: ParsedTicket,
        context: ConversationContext
    ) -> AgentResponse:
        ...
```

### 4.4 Confidence Scorer
**Purpose:** Determine if response is safe to send

**Signals:**

| Signal | Weight | How to Measure |
|--------|--------|----------------|
| Model uncertainty | 0.3 | Log probability of response tokens |
| Intent clarity | 0.2 | Router confidence score |
| Knowledge coverage | 0.25 | Retrieval similarity scores |
| Sentiment risk | 0.15 | Angry customer + uncertain = escalate |
| Complexity | 0.1 | Multi-part questions, edge cases |

**Thresholds:**
- `≥ 0.85` → Auto-respond
- `0.7 - 0.85` → Respond with "Let me know if this doesn't help"
- `< 0.7` → Escalate to human

### 4.5 Escalation Manager
**Purpose:** Hand off to humans with maximum context

**Context Package:**

```python
class EscalationContext:
    ticket: ParsedTicket
    conversation_history: list[Message]
    attempted_resolution: str  # what AI tried
    failure_reason: str  # why it's escalating
    customer_summary: CustomerProfile  # history, tier, sentiment
    suggested_actions: list[str]  # what human might try
    similar_resolved_tickets: list[TicketSummary]  # reference
    confidence_breakdown: dict  # why confidence was low
```

### 4.6 Feedback Loop
**Purpose:** Learn from human corrections

**Events to capture:**
- Human edited AI response → fine-tuning signal
- Human resolved after AI failed → add to knowledge base
- Customer marked "not helpful" → negative signal
- Customer marked "helpful" → positive signal

---

## 5. Tech Stack

### Recommended Stack (Portfolio-Ready)

| Component | Technology |
|-----------|------------|
| Orchestration | Python + LangGraph |
| LLM | Claude API (Haiku for routing, Sonnet for responses) |
| Vector DB | ChromaDB (local) or Pinecone (hosted) |
| Database | SQLite → PostgreSQL |
| Queue | Redis or in-memory |
| API | FastAPI |
| Frontend | React dashboard |

### Production-Grade Alternative

| Component | Technology |
|-----------|------------|
| Orchestration | Temporal + Python |
| LLM | Claude API with fallback to OpenAI |
| Vector DB | Pinecone or Weaviate |
| Database | PostgreSQL + TimescaleDB |
| Queue | RabbitMQ or SQS |
| API | FastAPI + GraphQL |
| Observability | LangSmith + Prometheus |

---

## 6. Data Models

```sql
-- Core tables
CREATE TABLE tickets (
    id UUID PRIMARY KEY,
    source VARCHAR(50),
    customer_id UUID,
    subject TEXT,
    body TEXT,
    intent VARCHAR(100),
    intent_confidence FLOAT,
    status VARCHAR(50),  -- open, in_progress, resolved, escalated
    assigned_to VARCHAR(100),  -- agent name or human
    created_at TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    ticket_id UUID REFERENCES tickets(id),
    role VARCHAR(20),  -- customer, agent, human
    content TEXT,
    confidence FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE escalations (
    id UUID PRIMARY KEY,
    ticket_id UUID REFERENCES tickets(id),
    reason TEXT,
    context_package JSONB,
    human_assignee VARCHAR(100),
    resolution TEXT,
    feedback_captured BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
);

CREATE TABLE feedback (
    id UUID PRIMARY KEY,
    ticket_id UUID REFERENCES tickets(id),
    type VARCHAR(50),  -- human_edit, customer_rating, resolution_used
    original_response TEXT,
    corrected_response TEXT,
    rating INT,
    created_at TIMESTAMP
);
```

---

## 7. API Design

```yaml
# Ticket Management
POST /tickets
  # Ingest new ticket from any source
  Request: { source, customer_id, subject, body, metadata }
  Response: { ticket_id, status, assigned_agent }

GET /tickets/{id}
  # Get ticket with full conversation history
  Response: { ticket, conversations, escalation_context }

POST /tickets/{id}/message
  # Customer sends follow-up message
  Request: { content }
  Response: { response, confidence, escalated }

POST /tickets/{id}/escalate
  # Force escalation to human
  Request: { reason }
  Response: { escalation_id, queue_position }

POST /tickets/{id}/resolve
  # Human marks ticket resolved
  Request: { resolution, feedback }
  Response: { success }

# Analytics
GET /analytics/summary
  Response: {
    tickets_today,
    auto_resolved_rate,
    avg_confidence,
    escalation_rate,
    avg_resolution_time
  }
```

---

## 8. Build Phases

### Phase 1: Core Loop
**Goal:** Single agent handles tickets end-to-end

- [ ] Ticket ingestion API
- [ ] Basic intent classification (LLM-based)
- [ ] Single generalist agent
- [ ] Simple confidence scoring (model logprobs)
- [ ] Response API
- [ ] SQLite persistence

**Demo:** Submit ticket via API, get AI response

---

### Phase 2: Specialization
**Goal:** Multi-agent routing with domain expertise

- [ ] Intent router with categories
- [ ] 3 specialist agents (billing, technical, account)
- [ ] Knowledge base per domain (seed with FAQ data)
- [ ] RAG integration for each agent
- [ ] Improved confidence scoring

**Demo:** Different tickets route to appropriate specialists

---

### Phase 3: Escalation
**Goal:** Graceful handoff to humans

- [ ] Escalation manager component
- [ ] Context package builder
- [ ] Human dashboard (view escalated tickets)
- [ ] Resolution capture flow
- [ ] Similar ticket search

**Demo:** Low-confidence ticket escalates with full context

---

### Phase 4: Learning Loop
**Goal:** System improves from feedback

- [ ] Feedback capture (human edits, ratings)
- [ ] Resolution → knowledge base pipeline
- [ ] Analytics dashboard
- [ ] Confidence calibration based on outcomes

**Demo:** Show improvement metrics over time

---

### Phase 5: Polish
**Goal:** Portfolio-ready

- [ ] Multi-channel ingestion (email webhook, chat widget)
- [ ] Beautiful dashboard UI
- [ ] Documentation + README
- [ ] Blog post explaining architecture
- [ ] Live demo deployment (Railway/Render)

---

## 9. Success Metrics

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Auto-resolution rate | > 60% | Shows the system works |
| Escalation with context | 100% | Never dead-end a customer |
| Avg confidence accuracy | > 85% | Calibrated uncertainty |
| Response latency p95 | < 3s | Production-viable |
| Cost per ticket | < $0.05 | Economically sensible |

---

## 10. Portfolio Presentation

### What makes this stand out:

1. **Real problem** - Every company has support costs
2. **Production patterns** - Confidence scoring, escalation, feedback loops
3. **Demonstrates depth** - Not just "call an LLM"
4. **Measurable** - Clear metrics to show in interviews
5. **Extensible** - Obvious paths to add features
6. **Live demo** - Recruiters can actually try it

### Interview talking points:

- Why confidence scoring matters (avoiding hallucination in production)
- How the feedback loop enables continuous improvement
- Tradeoffs between auto-resolution rate and customer satisfaction
- Cost optimization strategies (Haiku for routing, Sonnet for responses)
- How to handle edge cases and unknown intents
