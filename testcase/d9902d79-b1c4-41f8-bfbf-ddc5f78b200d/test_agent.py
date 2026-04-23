
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from agent import MeetingNotesSummarizerAgent, MeetingNotesRequest, MeetingNotesResponse

@pytest.mark.asyncio
async def test_integration_end_to_end_summarization_and_email_flow():
    """
    Integration test: End-to-End Summarization and Email Flow.
    Tests the full workflow from input normalization, preprocessing, LLM summary generation,
    action item extraction, formatting, consent validation, to email sending.
    """
    agent = MeetingNotesSummarizerAgent()

    # Prepare input
    req = MeetingNotesRequest(
        input_type="text",
        input_value="Alice: Let's schedule the next meeting for Friday.\nBob: I'll send the agenda.\nAttendees: Alice, Bob, Carol",
        summary_length="detailed",
        user_email="alice@example.com",
        participant_emails=["bob@example.com", "carol@example.com"],
        user_consent=True
    )

    # Mocks for each service
    with patch.object(agent.input_handler, "receive_input", return_value="Normalized transcript") as mock_receive_input, \
         patch.object(agent.preprocessor, "normalize_text", return_value="Cleaned transcript") as mock_normalize_text, \
         patch.object(agent.llm_service, "generate_summary", new_callable=AsyncMock, return_value="Meeting Overview: ...\nAttendees: Alice, Bob, Carol") as mock_generate_summary, \
         patch.object(agent.llm_service, "extract_action_items", new_callable=AsyncMock, return_value=[
             {"action": "Send agenda", "owner": "Bob", "due_date": "Friday", "priority": "High"}
         ]) as mock_extract_action_items, \
         patch.object(agent.summary_formatter, "format_summary", return_value="Formatted summary email body") as mock_format_summary, \
         patch.object(agent.consent_manager, "validate_consent", return_value=True) as mock_validate_consent, \
         patch.object(agent.email_sender, "send_email", return_value=True) as mock_send_email, \
         patch.object(agent.audit_logger, "log_event") as mock_log_event:

        response = await agent.process_meeting_notes(req)

        # Success criteria assertions
        mock_receive_input.assert_called_once_with("text", req.input_value)
        mock_normalize_text.assert_called_once_with("Normalized transcript")
        mock_generate_summary.assert_awaited_once_with("Cleaned transcript", "detailed")
        mock_extract_action_items.assert_awaited_once_with("Cleaned transcript")
        mock_format_summary.assert_called_once_with(
            "Meeting Overview: ...\nAttendees: Alice, Bob, Carol",
            [
                {"action": "Send agenda", "owner": "Bob", "due_date": "Friday", "priority": "High"}
            ],
            ["Alice", "Bob", "Carol"],
            "detailed"
        )
        mock_validate_consent.assert_called_once_with(True)
        mock_send_email.assert_called_once_with(
            "Formatted summary email body",
            ["bob@example.com", "carol@example.com"],
            "alice@example.com"
        )
        # Check MeetingNotesResponse fields
        assert isinstance(response, MeetingNotesResponse)
        assert response.success is True
        assert response.summary == "Formatted summary email body"
        assert response.action_items == [
            {"action": "Send agenda", "owner": "Bob", "due_date": "Friday", "priority": "High"}
        ]
        assert response.attendees == ["Alice", "Bob", "Carol"]
        assert response.email_status == "sent"
        assert response.error is None

        # Audit log events should be called at least once
        assert mock_log_event.call_count >= 1