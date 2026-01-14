# Support Escalation Agent

AI-powered customer support agent with intelligent escalation. Handles tier-1 support, knows its limits, and escalates to humans with full context when confidence is low.

## Features

- **Intent Classification**: Automatically categorizes tickets (billing, technical, account, etc.)
- **Confidence Scoring**: Multi-signal confidence calculation to determine when to escalate
- **Graceful Escalation**: Never dead-ends customers; always hands off with full context
- **Conversation Tracking**: Maintains full conversation history per ticket
- **Analytics**: Track auto-resolution rates, confidence scores, and escalation patterns

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the server

```bash
python run.py
```

Server starts at `http://localhost:8000`

## API Endpoints

### Create Ticket
```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "source": "api",
    "customer_id": "cust_123",
    "subject": "Cannot log in to my account",
    "body": "I have been trying to log in but keep getting an error message."
  }'
```

### Send Follow-up Message
```bash
curl -X POST http://localhost:8000/tickets/{ticket_id}/message \
  -H "Content-Type: application/json" \
  -d '{"content": "I tried that but it still does not work"}'
```

### Get Ticket Details
```bash
curl http://localhost:8000/tickets/{ticket_id}
```

### View Analytics
```bash
curl http://localhost:8000/analytics/summary
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Ticket Parser  │────▶│  Intent Router  │────▶│  Support Agent  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │  Confidence Scorer  │
                                              └──────────┬──────────┘
                                                         │
                                          ┌──────────────┴──────────────┐
                                          ▼                             ▼
                                    [Auto-respond]               [Escalate]
```

## Project Structure

```
support-escalation-agent/
├── src/
│   ├── agents/
│   │   └── support_agent.py    # Main agent logic
│   ├── api/
│   │   └── main.py             # FastAPI application
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models
│   │   └── ticket.py           # Pydantic models
│   └── services/
│       ├── confidence_scorer.py
│       └── intent_classifier.py
├── tests/
├── DESIGN.md                   # Full technical design
├── pyproject.toml
└── run.py
```

## Confidence Scoring

The agent uses multiple signals to determine confidence:

| Signal | Weight | Description |
|--------|--------|-------------|
| Intent Clarity | 25% | How clearly the ticket maps to a known category |
| Response Certainty | 35% | Language uncertainty markers in generated response |
| Sentiment Risk | 20% | Negative sentiment increases escalation likelihood |
| Complexity | 20% | Multi-part questions reduce confidence |

**Thresholds:**
- ≥ 0.85: Auto-respond
- 0.70-0.85: Respond with caveat
- < 0.70: Escalate to human

## Next Steps (Phase 2+)

- [ ] Specialist agents (billing, technical, account)
- [ ] RAG integration with knowledge base
- [ ] Feedback loop for continuous learning
- [ ] Human dashboard for escalated tickets
- [ ] Multi-channel ingestion (email, Slack)
