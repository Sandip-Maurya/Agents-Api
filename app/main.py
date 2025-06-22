# app/main.py

# fastapi import
from fastapi import FastAPI, HTTPException, Request

# Custom imports
from app.schema import AskRequest, AskResponse, VerifyRequest, VerifyResponse
from app.utils import extract_tool_info
from app.agents import guard_agent, model_agent
from app.handlers import ExceptionAndLoggingHandler

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ====================
# Logging Configuration
# ====================
import logging
logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")

# ====================
# FastAPI Initialization
# ====================
app = FastAPI(
    title="Guarded F1 & SHAP API",
    description="An API exposing two agents: a guardrail agent to validate requests and a model agent to compute F1 scores or generate SHAP plots for specified datasets.",
    version="1.0.0",
)

# register global exception handlers & middleware
ExceptionAndLoggingHandler.register(app)


@app.get("/", summary="Health check", tags=["Health"])
def root():
    """
    Simple health check endpoint.
    """
    return {"message": "API is up and running."}


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the agents for F1 or SHAP",
    tags=["Agent Interaction"],
    description="""
1. First the _guard_ agent filters your request.  
2. If it's valid, the model agent runs your query under MCP to compute the F1 or generate a SHAP plot.
3. The available data_sets are: `digits`, `iris`, `wine`, and `breast_cancer`.
"""
)
async def ask(req: AskRequest):
    """
    Passes `user_input` and `data_set` through:
     1. Guardrail agent
     2. Model agent (if allowed)
    """

    # 1) Guardrail check
    guard_out = await guard_agent.run(user_prompt=req.user_input)
    guard = guard_out.output

    if guard.type != "proceed":
        logger.info(f"Guard response: {guard.type}")
        return AskResponse(
            agent_response=guard.message,
            agent_name="guard",
        )

    # 2) Model agent under MCP
    async with model_agent.run_mcp_servers():
        model_out = await model_agent.run(user_prompt=req.user_input, deps=req.data_set)

    tool_used, tool_output = extract_tool_info(model_out)
    logger.info(f"Model used tool: {tool_used}")

    return AskResponse(
        agent_response=model_out.output,
        agent_name="model",
        tool_used=tool_used,
        tool_output=tool_output,
    )

@app.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a user query",
    tags=["Guard Validation"],
    description="""
Use the guard agent to check whether a given user_input + data_set
combination is acceptable.  
- If `is_valid` is `true`, you may safely call `/ask`.  
- Otherwise, inspect `message` for the guard'ss reason.
"""
)
async def verify(req: VerifyRequest, request: Request):
    """
    Runs only the guard agent and returns its verdict.
    """
    try:
        guard_out = await guard_agent.run(
            user_prompt=req.user_input,
        )

        guard = guard_out.output
        is_valid = (guard.type == "proceed")

        return VerifyResponse(is_valid=is_valid, message=guard.message)
    
    except Exception as e:
        logger.exception("Verification failed")
        # Turn any agent‐side error into a 502 Bad Gateway
        raise HTTPException(status_code=502, detail=f"Verification failed: {str(e)}")

