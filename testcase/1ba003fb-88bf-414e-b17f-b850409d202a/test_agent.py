
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
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
        input_value="Alice: Let's review the Q2 roadmap. Bob: I'll send the updated slides by Friday. Attendees: Alice, Bob, Carol.",
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
             {"action": "Send updated slides", "owner": "Bob", "due_date": "Friday", "priority": "High"}
         ]) as mock_extract_action_items, \
         patch.object(agent.summary_formatter, "format_summary", return_value="Formatted summary email body") as mock_format_summary, \
         patch.object(agent.consent_manager, "validate_consent", return_value=True) as mock_validate_consent, \
         patch.object(agent.email_sender, "send_email", return_value=True) as mock_send_email, \
         patch.object(agent.audit_logger, "log_event", wraps=agent.audit_logger.log_event) as mock_log_event:

        result = await agent.process_meeting_notes(req)

        # Assert result is MeetingNotesResponse and fields are populated
        assert isinstance(result, MeetingNotesResponse)
        assert result.success is True
        assert result.summary == "Formatted summary email body"
        assert result.action_items == [
            {"action": "Send updated slides", "owner": "Bob", "due_date": "Friday", "priority": "High"}
        ]
        assert result.attendees == ["Alice", "Bob", "Carol"]
        assert result.email_status == "sent"
        assert result.error is None

        # Service call assertions
        mock_receive_input.assert_called_once_with("text", req.input_value)
        mock_normalize_text.assert_called_once_with("Normalized transcript")
        mock_generate_summary.assert_awaited_once_with("Cleaned transcript", "detailed")
        mock_extract_action_items.assert_awaited_once_with("Cleaned transcript")
        mock_format_summary.assert_called_once_with(
            "Meeting Overview: ...\nAttendees: Alice, Bob, Carol",
            [
                {"action": "Send updated slides", "owner": "Bob", "due_date": "Friday", "priority": "High"}
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

        # AuditLogger.log_event called for each major event (at least 2: email_sent and email_not_sent)
        # We expect at least "email_sent" event
        found_email_sent = any(
            ((call.args and call.args[0] == "email_sent") or "email_sent" in (call.kwargs or {}).values()) or call.kwargs.get("event_type") == "email_sent"
            for call in mock_log_event.call_args_list
        )
        assert found_email_sent

        # Also check that at least one other event is logged (e.g., input_error, preprocessing_error, etc. not expected here)
        assert mock_log_event.call_count >= 1