# tests/test_shap.py

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

def test_shap_plot():
    resp = client.post("/ask", json={
        "user_input": "Please generate a SHAP plot for the iris dataset.",
        "data_set": "iris"
    })
    assert resp.status_code == 200
    data = resp.json()
    # Now the model agent must handle it
    assert data["agent_name"] == "model"
    assert data["tool_used"] == "mcp_shap_summary_tool"
    # tool_output should be a valid base64 string
    assert isinstance(data["tool_output"], str)
    assert len(data["tool_output"]) > 1000  # Ensure it's a reasonable length for an image
