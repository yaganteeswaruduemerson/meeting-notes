
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from agent import app, MeetingNotesRequest, MeetingNotesResponse
import agent
import httpx

@pytest.mark.asyncio
async def test_functional_consent_not_given_prevents_email_sending():
    """
    Ensures that if user_consent is False, the summary is generated but email is not sent and email_status is 'not_sent'.
    """
    # Prepare valid MeetingNotesRequest with user_consent=False, valid participant_emails and user_email
    req = {
        "input_type": "text",
        "input_value": "Attendees: Alice, Bob\nAction: Review Q2 budget\n",
        "summary_length": "detailed",
        "user_email": "user@example.com",
        "participant_emails": ["alice@example.com", "bob@example.com"],
        "user_consent": False
    }

    # Patch LLMService.generate_summary and extract_action_items to avoid real LLM calls
    fake_summary = "Meeting Overview: Discussed Q2 budget.\nAttendees: Alice, Bob"
    fake_action_items = [
        {"action": "Review Q2 budget", "owner": "Alice", "due_date": "2024-07-01", "priority": "High"}
    ]
    with patch.object(agent.MeetingNotesSummarizerAgent, "llm_service", create=True) as mock_llm_service, \
         patch.object(agent.MeetingNotesSummarizerAgent, "consent_manager", create=True) as mock_consent_manager, \
         patch.object(agent.MeetingNotesSummarizerAgent, "email_sender", create=True) as mock_email_sender:
        # Patch the agent's llm_service methods
        agent.agent.llm_service.generate_summary = AsyncMock(return_value=fake_summary)
        agent.agent.llm_service.extract_action_items = AsyncMock(return_value=fake_action_items)
        # Patch consent_manager.validate_consent to always return False
        agent.agent.consent_manager.validate_consent = MagicMock(return_value=False)
        # Patch email_sender.send_email to ensure it is NOT called
        agent.agent.email_sender.send_email = MagicMock()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/summarize", json=req)
            assert response.status_code == 200
            data = response.json()
            # Validate MeetingNotesResponse fields
            assert data["success"] is True
            assert data["summary"] is not None
            assert data["email_status"] == "not_sent"
            # Ensure send_email was NOT called
            assert not agent.agent.email_sender.send_email.called