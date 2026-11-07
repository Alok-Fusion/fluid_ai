import logging

from pydantic import ValidationError

from app.llm import LLMClient, LLMProviderError, extract_json
from app.schemas import PlanTask, SectionContent, TableContent

logger = logging.getLogger(__name__)

WRITE_SECTION_SYSTEM_PROMPT = """You write a document section. Return strict JSON with keys:
- "title": section title (string)
- "paragraphs": list of paragraphs (strings, exactly 1 or 2 short paragraphs)
- "bullet_points": list of bullet points (strings, at most 5 bullets, empty if none needed)

Keep content professional, concrete, and concise: limit to 2 short paragraphs and at most 5 bullets. Do not include markdown code fences or conversational text outside JSON."""

MAKE_TABLE_SYSTEM_PROMPT = """You generate a structured table for a document. Return strict JSON with keys:
- "title": table section title (string)
- "description": introductory note (string). If numbers, timelines, or metrics are estimated or mock, explicitly label them as illustrative.
- "headers": list of column header strings
- "rows": list of row items, where each item is a list of strings matching headers length (at most 6 rows)

Keep data realistic, precise, and concise: limit the table to at most 6 rows. Do not include markdown code fences or conversational text outside JSON."""


async def write_section(
    client: LLMClient,
    task: PlanTask,
    original_request: str,
    assumptions: list[str],
    context_summary: str,
) -> tuple[SectionContent, str]:
    """Execute the write_section tool to produce text paragraphs and bullet points."""
    user_prompt = (
        f"Original Business Request: {original_request}\n\n"
        f"Assumptions:\n" + "\n".join(f"- {a}" for a in assumptions) + "\n\n"
        f"Prior Sections Summary: {context_summary or 'None'}\n\n"
        f"Task #{task.id}: {task.title}\n"
        f"Goal: {task.goal}\n\n"
        f"Keep the section to 2 short paragraphs and at most 5 bullets. Return JSON adhering to the specified schema."
    )

    last_error: Exception | None = None
    for attempt in range(2):
        prompt = (
            user_prompt
            if attempt == 0
            else f"{user_prompt}\n\nPrevious attempt failed with error: {last_error}. Return valid JSON."
        )
        response = await client.generate(WRITE_SECTION_SYSTEM_PROMPT, prompt)
        try:
            data = extract_json(response.text)
            content = SectionContent.model_validate(data)
            return content, response.provider
        except (ValueError, ValidationError, LLMProviderError) as exc:
            last_error = exc
            logger.warning(
                "Task %d write_section attempt %d failed validation: %s",
                task.id,
                attempt + 1,
                exc,
            )

    raise ValueError(f"Task {task.id} write_section failed after retries: {last_error}")


async def make_table(
    client: LLMClient,
    task: PlanTask,
    original_request: str,
    assumptions: list[str],
    context_summary: str,
) -> tuple[TableContent, str]:
    """Execute the make_table tool to produce structured tabular data."""
    user_prompt = (
        f"Original Business Request: {original_request}\n\n"
        f"Assumptions:\n" + "\n".join(f"- {a}" for a in assumptions) + "\n\n"
        f"Prior Sections Summary: {context_summary or 'None'}\n\n"
        f"Task #{task.id}: {task.title}\n"
        f"Goal: {task.goal}\n\n"
        f"Keep the table to at most 6 rows. Return JSON with title, description (explicitly noting illustrative data if estimated), headers, and rows."
    )

    last_error: Exception | None = None
    for attempt in range(2):
        prompt = (
            user_prompt
            if attempt == 0
            else f"{user_prompt}\n\nPrevious attempt failed with error: {last_error}. Return valid JSON."
        )
        response = await client.generate(MAKE_TABLE_SYSTEM_PROMPT, prompt)
        try:
            data = extract_json(response.text)
            content = TableContent.model_validate(data)
            return content, response.provider
        except (ValueError, ValidationError, LLMProviderError) as exc:
            last_error = exc
            logger.warning(
                "Task %d make_table attempt %d failed validation: %s",
                task.id,
                attempt + 1,
                exc,
            )

    raise ValueError(f"Task {task.id} make_table failed after retries: {last_error}")

