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

def test_f1_score_live():
    resp = client.post("/ask", json={
        "user_input": "What is the F1 score of the model?",
        "data_set": "digits"
    })
    assert resp.status_code == 200
    data = resp.json()
    # Now the model agent must handle it
    assert data["agent_name"] == "model"
    assert data["tool_used"] == "mcp_f1_score_tool"
    # tool_output should be a float
    assert isinstance(float(data["tool_output"]), float)
