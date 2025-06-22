import sys
import os

# 1) Figure out the project root (one level up from tests/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 2) Prepend to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from fastapi.testclient import TestClient

from app.main import app  # adjust if your FastAPI instance lives elsewhere

client = TestClient(app)

def test_greeting():
    response = client.post(
        "/ask",
        json={"user_input": "Hello there!", "data_set": "iris"}
    )
    assert response.status_code == 200
    data = response.json()

    # Guardrail should catch greetings
    assert data["agent_name"] == "guard"
    assert data["tool_used"] is None
    assert data["tool_output"] is None


def test_invalid_question():
    response = client.post(
        "/ask",
        json={"user_input": "What is the capital of France?", "data_set": "wine"}
    )
    assert response.status_code == 200
    data = response.json()

    # Guardrail should reject unrelated queries
    assert data["agent_name"] == "guard"
    assert data["tool_used"] is None
    assert data["tool_output"] is None
