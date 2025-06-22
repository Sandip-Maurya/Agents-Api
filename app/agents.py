# app/agents.py

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStreamableHTTP

from app.instructions import guard_instructions, model_instructions
from app.schema import GuardResult

import logging
logger = logging.getLogger(__name__)

# ====================
# Agent Setup
# ====================

# 1) Guardrail agent


try:
    model_name = OpenAIModel("gpt-4o-mini")
    mcp_server = MCPServerStreamableHTTP("http://localhost:8000/mcp/")

    guard_agent = Agent(
        model_name,
        instructions=guard_instructions,
        output_type=GuardResult,
    )

    # 2) Model agent with MCP support

    model_agent = Agent(
        model_name, # can be same as guard_agent or different
        instructions=model_instructions,
        mcp_servers=[mcp_server],
        deps_type=str,
    )

except Exception as e:
    logger.exception("Failed to initialize agent")
    # Escalate so app won’t start with a broken agent
    raise RuntimeError("Agent configuration error") from e


@model_agent.system_prompt(dynamic=True)
def inject_dataset(ctx: RunContext[str]) -> str:
    """Injects the dataset name into the system prompt."""
    return f"Use dataset: {ctx.deps}. Directory name matches dataset name."

