# Support Escalation Agent

Multi-agent customer support system with specialist routing, RAG-enhanced responses, and intelligent escalation. Routes tickets to domain experts (billing, technical, account), retrieves relevant knowledge base content, and escalates to humans when confidence is low.

## Features

- **Multi-Agent Routing**: Automatically routes tickets to specialist agents based on intent
- **Specialist Agents**: Billing, Technical, and Account agents with domain-specific prompts
- **RAG Integration**: ChromaDB vector store with FAQ content enhances agent responses
- **Intent Classification**: Claude Haiku-based classification into 12 categories
- **Confidence Scoring**: Multi-signal scoring (intent clarity, response certainty, sentiment, complexity)
- **Graceful Escalation**: Context packaging with suggested actions for human agents
- **Conversation Tracking**: Full history per ticket with SQLite persistence
- **Analytics**: Resolution rates, confidence scores, routing distribution

## Quick Start

1. Install: `pip install -e .`
2. Configure: `cp .env.example .env` and add ANTHROPIC_API_KEY
3. Seed KB: `python scripts/seed_knowledge_base.py`
4. Run: `python run.py` (starts at http://localhost:8000)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /tickets | POST | Create ticket, get AI response |
| /tickets/{id} | GET | Get ticket with conversation history |
| /tickets/{id}/message | POST | Send follow-up message |
| /tickets/{id}/escalate | POST | Force escalation to human |
| /tickets/{id}/resolve | POST | Mark ticket resolved |
| /analytics/summary | GET | Get system metrics |
| /knowledge/stats | GET | Knowledge base statistics |

## Specialist Agents

| Agent | Handles | Auto-Escalates On |
|-------|---------|-------------------|
| Billing | Charges, refunds, subscriptions | Fraud, legal threats, chargebacks |
| Technical | Bugs, how-to, features | Data loss, security issues, outages |
| Account | Login, 2FA, deletion | Hacked accounts, GDPR, legal requests |

## Confidence Scoring

| Signal | Weight |
|--------|--------|
| Intent Clarity | 25% |
| Response Certainty | 35% |
| Sentiment Risk | 20% |
| Complexity | 20% |

Thresholds: >=0.85 auto-respond | 0.70-0.85 respond with caveat | <0.70 escalate

## Known Issues (TODO)

- [ ] RAG retrieval not fully wired to specialist agents
- [ ] Phase 2 tests incomplete  
- [ ] Feedback loop not implemented
- [ ] No human dashboard UI

## Tech Stack

FastAPI | Claude (Haiku + Sonnet) | ChromaDB | SQLite | sentence-transformers
