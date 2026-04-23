
import pytest
from agent import MeetingNotesSummarizerAgent, FollowUpQueryRequest, FollowUpQueryResponse

def test_integration_followup_query_with_missing_context():
    """
    Integration test: Checks that if transcript_context is empty, the agent returns
    a FollowUpQueryResponse with success=False and an error message.
    """
    agent = MeetingNotesSummarizerAgent()
    # Step 1: Try to construct FollowUpQueryRequest with empty transcript_context (should raise ValueError)
    with pytest.raises(ValueError) as excinfo:
        FollowUpQueryRequest(query_text="What did Alice agree to do?", transcript_context="")
    assert "transcript_context cannot be empty" in str(excinfo.value)

    # Step 2: If for some reason the validation is bypassed, agent should still handle it gracefully
    # (simulate a direct call with empty context)
    response = pytest.run(
        agent.answer_follow_up_query("What did Alice agree to do?", "")
    ) if hasattr(pytest, "run") else None
    # If pytest.run is not available, fallback to asyncio
    if response is None:
        import asyncio
        response = asyncio.run(agent.answer_follow_up_query("What did Alice agree to do?", ""))
    assert isinstance(response, FollowUpQueryResponse)
    assert response.success is False
    assert response.error is not None
    assert "error" in response.error.lower() or "unable" in response.error.lower()