
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient

import agent

@pytest.fixture
def valid_meeting_notes_request():
    """Fixture for a valid MeetingNotesRequest payload."""
    return {
        "input_type": "text",
        "input_value": (
            "Attendees: Alice, Bob, Carol\n"
            "Discussion: Project timeline and deliverables.\n"
            "Decisions Made: Move deadline to next Friday.\n"
            "Action Items: Bob to update the project plan by Wednesday. Carol to notify stakeholders.\n"
            "Next Steps: Schedule follow-up meeting."
        ),
        "summary_length": "detailed",
        "user_email": "alice@example.com",
        "participant_emails": ["bob@example.com", "carol@example.com"],
        "user_consent": True
    }

@pytest.fixture
def mock_llm_summary():
    """Returns a realistic LLM summary string with required sections."""
    return (
        "Meeting Overview:\nProject timeline and deliverables discussed.\n\n"
        "Key Discussion Points:\n- Timeline adjustment\n- Deliverables review\n\n"
        "Decisions Made:\n- Deadline moved to next Friday\n\n"
        "Action Items:\n- Bob to update the project plan by Wednesday (Owner: Bob, Due: Wednesday, Priority: High)\n"
        "- Carol to notify stakeholders (Owner: Carol, Due: Not specified, Priority: Medium)\n\n"
        "Next Steps:\n- Schedule follow-up meeting\n\n"
        "Attendees: Alice, Bob, Carol"
    )

@pytest.fixture
def mock_action_items():
    """Returns a list of action items as would be extracted by the LLM."""
    return [
        {
            "action": "Update the project plan",
            "owner": "Bob",
            "due_date": "Wednesday",
            "priority": "High"
        },
        {
            "action": "Notify stakeholders",
            "owner": "Carol",
            "due_date": "Not specified",
            "priority": "Medium"
        }
    ]

@pytest.fixture
def test_client():
    """Fixture for FastAPI TestClient."""
    return TestClient(agent.app)

@pytest.mark.asyncio
def test_functional_summarize_endpoint_returns_structured_summary(
    test_client,
    valid_meeting_notes_request,
    mock_llm_summary,
    mock_action_items
):
    """
    Validates that the /summarize endpoint returns a structured summary, action items,
    and attendees when provided with a valid MeetingNotesRequest.
    """
    # Patch LLMService.generate_summary, extract_action_items, and EmailSender.send_email
    with patch.object(agent.agent.llm_service, "generate_summary", new=AsyncMock(return_value=mock_llm_summary)), \
         patch.object(agent.agent.llm_service, "extract_action_items", new=AsyncMock(return_value=mock_action_items)), \
         patch.object(agent.agent.email_sender, "send_email", return_value=True):

        response = test_client.post("/summarize", json=valid_meeting_notes_request)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["summary"] is not None
        # Check that required sections are present in the summary
        for section in [
            "Meeting Overview",
            "Key Discussion Points",
            "Decisions Made",
            "Action Items",
            "Next Steps",
            "Attendees"
        ]:
            assert section in data["summary"]
        assert isinstance(data["action_items"], list)
        assert isinstance(data["attendees"], list)
        assert data["email_status"] == "sent"

    # Error scenario: LLMService.generate_summary returns FALLBACK_RESPONSE
    with patch.object(agent.agent.llm_service, "generate_summary", new=AsyncMock(return_value=agent.FALLBACK_RESPONSE)), \
         patch.object(agent.agent.llm_service, "extract_action_items", new=AsyncMock(return_value=mock_action_items)), \
         patch.object(agent.agent.email_sender, "send_email", return_value=True):

        response = test_client.post("/summarize", json=valid_meeting_notes_request)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    # Error scenario: LLMService.extract_action_items returns empty list
    with patch.object(agent.agent.llm_service, "generate_summary", new=AsyncMock(return_value=mock_llm_summary)), \
         patch.object(agent.agent.llm_service, "extract_action_items", new=AsyncMock(return_value=[])), \
         patch.object(agent.agent.email_sender, "send_email", return_value=True):

        response = test_client.post("/summarize", json=valid_meeting_notes_request)
        assert response.status_code == 200
        data = response.json()
        # Should still succeed, but action_items is empty
        assert data["success"] is True
        assert isinstance(data["action_items"], list)
        assert data["email_status"] == "sent"

    # Error scenario: EmailSender.send_email returns False
    with patch.object(agent.agent.llm_service, "generate_summary", new=AsyncMock(return_value=mock_llm_summary)), \
         patch.object(agent.agent.llm_service, "extract_action_items", new=AsyncMock(return_value=mock_action_items)), \
         patch.object(agent.agent.email_sender, "send_email", return_value=False):

        response = test_client.post("/summarize", json=valid_meeting_notes_request)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["email_status"] == "failed"