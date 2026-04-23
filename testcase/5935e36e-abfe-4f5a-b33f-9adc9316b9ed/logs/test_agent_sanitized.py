
import pytest
from unittest.mock import patch, MagicMock
from agent import MeetingNotesSummarizerAgent, MeetingNotesRequest, MeetingNotesResponse

@pytest.mark.asyncio
async def test_integration_error_handling_on_invalid_input_type():
    """
    Verifies that an unsupported input_type in MeetingNotesRequest triggers error handling and returns
    a MeetingNotesResponse with success=False and appropriate error message.
    """
    # Prepare input with invalid input_type
    req = MeetingNotesRequest(
        input_type="invalid_type",
        input_value="This is a valid transcript.",
        summary_length="detailed",
        user_email="user@example.com",
        participant_emails=["a@example.com", "b@example.com"],
        user_consent=True
    )
    agent_instance = MeetingNotesSummarizerAgent()

    # Patch InputHandler.receive_input to simulate error (should return "")
    with patch.object(agent_instance.input_handler, "receive_input", return_value="") as mock_receive_input, \
         patch.object(agent_instance.error_handler, "handle_error", wraps=agent_instance.error_handler.handle_error) as mock_handle_error, \
         patch.object(agent_instance.audit_logger, "log_event") as mock_log_event:
        result = await agent_instance.process_meeting_notes(req)

        # InputHandler.receive_input should be called and return ""
        mock_receive_input.assert_called_once_with("invalid_type", "This is a valid transcript.")

        # ErrorHandler.handle_error should be called with "GENERIC_ERROR"
        assert any(
            (((call.args and call.args[0] == "GENERIC_ERROR") or "GENERIC_ERROR" in (call.kwargs or {}).values()) or "GENERIC_ERROR" in (call.kwargs or {}).values())
            for call in mock_handle_error.call_args_list
        )

        # Should return MeetingNotesResponse with success=False and error message
        assert isinstance(result, MeetingNotesResponse)
        assert result.success is False
        assert result.error is not None
        assert "Input normalization failed" in result.error or "Unable to generate a complete summary" in result.error

        # AuditLogger.log_event should be called for input_error
        assert any(
            (((call.args and call.args[0] == "input_error") or "input_error" in (call.kwargs or {}).values()) or "input_error" in (call.kwargs or {}).values())
            for call in mock_log_event.call_args_list
        )