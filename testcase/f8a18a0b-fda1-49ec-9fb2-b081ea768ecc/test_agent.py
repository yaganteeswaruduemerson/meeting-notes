
import pytest
from unittest.mock import patch, MagicMock
from agent import MeetingNotesSummarizerAgent, MeetingNotesRequest, MeetingNotesResponse

@pytest.mark.asyncio
async def test_integration_error_handling_on_invalid_input_type():
    """
    Verifies that an unsupported input_type in MeetingNotesRequest triggers error handling and returns
    MeetingNotesResponse with success=False and appropriate error message.
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

    # Patch InputHandler.receive_input to simulate ValueError and return ""
    with patch.object(agent_instance.input_handler, "receive_input", return_value="") as mock_receive_input, \
         patch.object(agent_instance.error_handler, "handle_error", wraps=agent_instance.error_handler.handle_error) as mock_handle_error, \
         patch.object(agent_instance.audit_logger, "log_event", wraps=agent_instance.audit_logger.log_event) as mock_log_event:

        response = await agent_instance.process_meeting_notes(req)

        # InputHandler.receive_input returns empty string
        mock_receive_input.assert_called_once_with("invalid_type", "This is a valid transcript.")

        # ErrorHandler.handle_error is called with 'GENERIC_ERROR'
        assert any(
            ((call.args and call.args[0] == "GENERIC_ERROR") or "GENERIC_ERROR" in (call.kwargs or {}).values())
            for call in mock_handle_error.call_args_list
        )

        # AuditLogger.log_event is called with 'input_error'
        assert any(
            ((call.args and call.args[0] == "input_error") or "input_error" in (call.kwargs or {}).values())
            for call in mock_log_event.call_args_list
        )

        # response.success is False
        assert isinstance(response, MeetingNotesResponse)
        assert response.success is False

        # response.error contains 'Input normalization failed'
        assert response.error is not None
        assert "Input normalization failed" in response.error