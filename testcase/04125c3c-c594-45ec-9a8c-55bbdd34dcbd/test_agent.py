
import pytest
from starlette.testclient import TestClient
from agent import app

def test_integration_followup_query_with_missing_context():
    """
    Integration test: Checks that when transcript_context is empty,
    the /followup endpoint returns an error response.
    """
    client = TestClient(app)
    payload = {
        "query_text": "What did Alice agree to do?",
        "transcript_context": ""
    }
    response = client.post("/followup", json=payload)
    assert response.status_code in (422, 200)
    data = response.json()
    assert isinstance(data, dict)
    assert data.get("success") is False
    # Accept either validation error or fallback error message
    assert (
        ("transcript_context cannot be empty" in str(data.get("error", "")))
        or ("Malformed request" in str(data.get("error", "")))
        or ("fallback" in str(data.get("error", "")).lower())
        or ("insufficient" in str(data.get("error", "")).lower())
        or ("Please provide additional details" in str(data.get("error", "")).lower())
    )