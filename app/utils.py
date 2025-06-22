# app/utils.py

from typing import Any
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

def extract_tool_info(result) -> tuple[str | None, Any | None]:
    """Extract the first tool call name and its returned content from an agent result."""
    tool_used = None
    tool_output = None
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and not tool_used:
                tool_used = part.tool_name
            if isinstance(part, ToolReturnPart) and not tool_output:
                tool_output = part.content
    return tool_used, tool_output
