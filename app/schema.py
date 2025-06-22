# app/schema.py

from typing import Any, Literal
from pydantic import BaseModel, Field

# ====================
# Schemas
# ====================
class AskRequest(BaseModel):
    """
    Request body for /ask endpoint.

    - **user_input**: The question or command for F1 score or SHAP plot.
    - **data_set**: One of the supported dataset names.
    """
    user_input: str = Field(
        ..., 
        examples=[
            "What is the F1 score?",
            "Can you generate a SHAP plot for the iris dataset?"
        ],
        description="Question to the agent."
    )
    data_set: Literal["digits", "iris", "wine", "breast_cancer"]

class AskResponse(BaseModel):
    """
    Standard response from the /ask endpoint.

    - **agent_response**: The final textual response from the agent.
    - **agent_name**: Which agent handled the request ('guard' or 'model').
    - **tool_used**: The name of the tool the model agent called, if any.
    - **tool_output**: Raw output returned by the called tool, if any.
    """
    agent_response: str
    agent_name: Literal["guard", "model"]
    tool_used: str | None = Field(default=None, description="Name of the tool invoked by the model agent.")
    tool_output: Any | None = Field(default=None, description="Raw output from the tool.")

class GuardResult(BaseModel):
    '''
    Result from the guardrail agent.
    - **type**: Type of response, can be "greeting", "error", or "proceed".
    - **message**: The message to return to the user.
    '''

    type: Literal["greeting", "error", "proceed"]
    message: str

class VerifyRequest(BaseModel):
    """
    Payload to verify whether a user query + dataset combo
    is allowed by the guard agent.
    """
    user_input: str = Field(
        ...,
        examples=[
            "What is the F1 score?",
            "Generate a SHAP plot for iris."
        ],
        description="The user’s natural-language query."
    )
    data_set: Literal["digits", "iris", "wine", "breast_cancer"] = Field(
        ...,
        description="Which dataset to evaluate (must match one of the supported names)."
    )

class VerifyResponse(BaseModel):
    """
    Indicates whether the guard agent would allow this query.
    """
    is_valid: bool = Field(
        ...,
        description="True if the guard agent returns type='proceed', else False."
    )
    message: str = Field(
        ...,
        description="The guard agent’s message (error or greeting or empty)."
    )
