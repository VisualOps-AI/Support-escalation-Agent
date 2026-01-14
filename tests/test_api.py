import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from src.api.main import app
from src.models.ticket import AgentResponse


@pytest.fixture
def mock_agent_response():
    return AgentResponse(
        message="I'd be happy to help you with your login issue. Let me guide you through some troubleshooting steps.",
        confidence=0.85,
        intent="account.access_issue",
        should_escalate=False,
        escalation_reason=None,
        suggested_actions=[],
    )


@pytest.fixture
def mock_escalated_response():
    return AgentResponse(
        message="I understand you're having trouble. Let me connect you with a specialist.",
        confidence=0.55,
        intent="unknown",
        should_escalate=True,
        escalation_reason="unclear customer intent; low response confidence",
        suggested_actions=["Review full ticket history"],
    )


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_create_ticket(mock_agent_response):
    with patch("src.api.main.agent.handle_ticket", new_callable=AsyncMock) as mock_handle:
        mock_handle.return_value = mock_agent_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/tickets",
                json={
                    "source": "api",
                    "customer_id": "test_customer",
                    "subject": "Cannot log in",
                    "body": "I keep getting an error when trying to log in.",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "ticket_id" in data
            assert data["intent"] == "account.access_issue"
            assert data["escalated"] is False


@pytest.mark.asyncio
async def test_create_ticket_escalated(mock_escalated_response):
    with patch("src.api.main.agent.handle_ticket", new_callable=AsyncMock) as mock_handle:
        mock_handle.return_value = mock_escalated_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/tickets",
                json={
                    "source": "api",
                    "customer_id": "test_customer",
                    "subject": "Something weird",
                    "body": "asdfghjkl not sure what's happening",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["escalated"] is True
            assert data["assigned_agent"] == "pending_human"


@pytest.mark.asyncio
async def test_get_nonexistent_ticket():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/tickets/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_analytics_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_tickets" in data
        assert "auto_resolved_rate" in data
