import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, Response

from app.agent import AgentOrchestrator, PlanningError
from app.llm import LLMClient, LLMProviderError
from app.schemas import AgentRequest, AgentResponse, HealthResponse
from app.store import RunStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore above instructions",
    "disregard previous instructions",
    "disregard all previous",
    "disregard instructions",
    "forget previous instructions",
    "system prompt",
    "you are now in developer mode",
    "dan mode",
    "jailbreak",
    "bypass safety",
    "reveal your prompt",
    "show system prompt",
]

store = RunStore()
llm_client = LLMClient()
orchestrator = AgentOrchestrator(llm_client, store)


def validate_request_text(raw_text: str) -> str:
    """Validate request constraints and reject invalid or adversarial input."""
    text = raw_text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request text cannot be empty.",
        )
    if len(text) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request text is too short ({len(text)} characters). Minimum length is 10.",
        )
    if len(text) > 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request text exceeds maximum allowed length of 2000 characters ({len(text)} characters provided).",
        )

    lowered = text.lower()
    for phrase in PROMPT_INJECTION_PATTERNS:
        if phrase in lowered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request rejected: potentially adversarial instruction detected.",
            )
    return text


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ensure output directories exist on application startup."""
    os.makedirs("outputs", exist_ok=True)
    yield


app = FastAPI(
    title="Autonomous Agent Document Service",
    description="Generates structured Word documents from business requests using LLM orchestration.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
@app.get("/page", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the static demo UI page."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
    return FileResponse(html_path)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Quietly handle favicon requests."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verify service availability."""
    return HealthResponse(status="ok")


@app.post("/agent", response_model=AgentResponse)
async def run_agent(payload: AgentRequest) -> AgentResponse:
    """Process a natural language business request and produce a Word document."""
    validated_text = validate_request_text(payload.request)

    try:
        return await orchestrator.run(validated_text)
    except PlanningError as exc:
        logger.warning("Planning failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Planning error: {exc}",
        ) from exc
    except LLMProviderError as exc:
        logger.error("All LLM providers failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service failure: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during agent execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution error: {exc}",
        ) from exc


@app.get("/download/{run_id}")
async def download_document(run_id: str) -> FileResponse:
    """Download the generated Word document for a completed run."""
    record = store.get(run_id)
    if not record or not os.path.exists(record.docx_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document for run_id '{run_id}' not found.",
        )

    return FileResponse(
        path=record.docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{run_id}.docx",
    )

