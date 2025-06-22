# app/utils.py

from http.client import HTTPException
from typing import Any
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def extract_tool_info(result) -> tuple[str | None, Any | None]:
    """Extract the first tool call name and its returned content from an agent result."""
    try:
        tool_used = tool_output = None
        for msg in result.all_messages():
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and not tool_used:
                    tool_used = part.tool_name
                if isinstance(part, ToolReturnPart) and not tool_output:
                    tool_output = part.content
        return tool_used, tool_output
    
    except Exception as e:
        logger.exception("Error while extracting tool info from agent result")
        # wrap or re‐raise as HTTPException for endpoint to catch
        raise HTTPException(status_code=500, detail="Failed to parse agent tool output") from e
    