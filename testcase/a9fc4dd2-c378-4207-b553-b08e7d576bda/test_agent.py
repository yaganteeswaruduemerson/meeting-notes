
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from agent import app, MeetingNotesRequest, FALLBACK_RESPONSE
from agent import MeetingNotesResponse
import agent
import httpx

@pytest.mark.asyncio
async def test_functional_summarize_endpoint_returns_structured_summary():
    """
    Validates that the /summarize endpoint processes a valid MeetingNotesRequest and returns
    a MeetingNotesResponse with a structured summary, action items, attendees, and email_status='sent'.
    """
    # Prepare input
    transcript = (
        "Attendees: Alice, Bob, Carol\n"
        "Discussion: Project timeline and deliverables.\n"
        "Action: Bob to send updated project plan by Friday. Carol to review requirements."
    )
    request_data = {
        "input_type": "text",
        "input_value": transcript,
        "summary_length": "detailed",
        "user_email": "alice@example.com",
        "participant_emails": ["bob@example.com", "carol@example.com"],
        "user_consent": True
    }

    # Prepare expected LLMService.generate_summary and extract_action_items outputs
    fake_summary = (
        "Meeting Overview:\nProject timeline and deliverables discussed.\n\n"
        "Key Discussion Points:\n- Timeline\n- Deliverables\n\n"
        "Decisions Made:\n- Proceed with updated plan\n\n"
        "Action Items:\n- Bob to send updated project plan by Friday (Owner: Bob, Due: Friday, Priority: High)\n"
        "- Carol to review requirements (Owner: Carol, Due: Not specified, Priority: Medium)\n\n"
        "Attendees: Alice, Bob, Carol"
    )
    fake_action_items = [
        {
            "action": "Send updated project plan",
            "owner": "Bob",
            "due_date": "Friday",
            "priority": "High"
        },
        {
            "action": "Review requirements",
            "owner": "Carol",
            "due_date": "Not specified",
            "priority": "Medium"
        }
    ]

    # Patch LLMService.generate_summary, extract_action_items, and EmailSender.send_email
    with patch.object(agent.MeetingNotesSummarizerAgent, "llm_service", create=True) as mock_llm_service, \
         patch.object(agent.MeetingNotesSummarizerAgent, "email_sender", create=True) as mock_email_sender, \
         patch.object(agent.MeetingNotesSummarizerAgent, "input_handler", create=True) as mock_input_handler, \
         patch.object(agent.MeetingNotesSummarizerAgent, "preprocessor", create=True) as mock_preprocessor, \
         patch.object(agent.MeetingNotesSummarizerAgent, "summary_formatter", create=True) as mock_summary_formatter, \
         patch.object(agent.MeetingNotesSummarizerAgent, "consent_manager", create=True) as mock_consent_manager, \
         patch.object(agent.MeetingNotesSummarizerAgent, "error_handler", create=True) as mock_error_handler, \
         patch.object(agent.MeetingNotesSummarizerAgent, "audit_logger", create=True) as mock_audit_logger:

        # Setup mocks for the agent instance used by the FastAPI app
        agent.agent.llm_service.generate_summary = AsyncMock(return_value=fake_summary)
        agent.agent.llm_service.extract_action_items = AsyncMock(return_value=fake_action_items)
        agent.agent.email_sender.send_email = MagicMock(return_value=True)
        agent.agent.input_handler.receive_input = MagicMock(return_value=transcript)
        agent.agent.preprocessor.normalize_text = MagicMock(return_value=transcript)
        agent.agent.summary_formatter.format_summary = MagicMock(return_value=fake_summary)
        agent.agent.consent_manager.validate_consent = MagicMock(return_value=True)
        agent.agent.error_handler.handle_error = MagicMock(return_value=FALLBACK_RESPONSE)
        agent.agent.audit_logger.log_event = MagicMock()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/summarize", json=request_data)
            assert resp.status_code == 200
            data = resp.json()
            # Validate MeetingNotesResponse fields
            assert data["success"] is True
            assert data["summary"] is not None
            # Check for required sections in summary
            for section in [
                "Meeting Overview", "Key Discussion Points", "Decisions Made", "Action Items", "Attendees"
            ]:
                assert section in data["summary"]
            assert isinstance(data["action_items"], list)
            assert isinstance(data["attendees"], list)
            assert data["email_status"] == "sent"

            # Check that the mocked methods were called as expected
            agent.agent.llm_service.generate_summary.assert_awaited()
            agent.agent.llm_service.extract_action_items.assert_awaited()
            agent.agent.email_sender.send_email.assert_called_once()
            agent.agent.input_handler.receive_input.assert_called_once()
            agent.agent.preprocessor.normalize_text.assert_called_once()
            agent.agent.summary_formatter.format_summary.assert_called_once()
            agent.agent.consent_manager.validate_consent.assert_called_once()