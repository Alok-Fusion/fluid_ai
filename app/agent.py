import asyncio
import logging
import os
import uuid
from typing import Any

from pydantic import ValidationError

from app.docx_builder import build_docx
from app.llm import LLMClient, LLMProviderError, extract_json
from app.schemas import (
    AgentResponse,
    PlannerOutput,
    PlanTask,
    ReviewOutput,
    SectionContent,
    TableContent,
    TaskResult,
    TaskSummary,
)
from app.store import RunRecord, RunStore
from app.tools import make_table, write_section

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Raised when the planner fails to produce a valid plan."""


PLANNER_SYSTEM_PROMPT = """You are a document planner. Analyze the business request and define a structured document plan.
Return strict JSON with keys:
- "doc_type": one of "proposal", "meeting_minutes", "project_plan", "business_report", "technical_design", "sop", "product_spec"
- "title": descriptive title for the document
- "assumptions": list of strings detailing assumptions made for missing details or resolved conflicts
- "tasks": list of 5 to 9 tasks in document order, each with:
    - "id": integer starting from 1
    - "title": section title
    - "goal": concrete instructions for what this section must contain
    - "tool": "write_section" or "make_table"

If the request is ambiguous regarding doc_type, pick the most appropriate type.
If requirements conflict or numbers are missing, record an assumption and state the resolution.
Use make_table for timelines, budgets, risk matrices, milestones, or structured comparisons.
For project_plan and proposal documents, include at least one make_table task (timeline or budget).
Do not output markdown code fences or text outside the JSON."""

REVIEWER_SYSTEM_PROMPT = """You are a document reviewer. Review the assembled document sections against the user request and assumptions.
Return strict JSON with keys:
- "summary": 1 to 2 sentence summary of overall document quality and alignment
- "needs_fix": boolean indicating if critical revisions are needed
- "issues": list of issues found, each with:
    - "section_id": integer matching the task id
    - "issue": description of the problem
    - "suggested_fix": specific direction for fixing the section

Only set needs_fix to true if there are contradictions, missing core requirements, or unlabelled mock data.
Do not output markdown code fences or text outside the JSON."""

REVISE_SYSTEM_PROMPT = """You revise a document section to address a reviewer issue.
Return strict JSON adhering to the tool schema:
If write_section: {"title": "...", "paragraphs": ["..."], "bullet_points": ["..."]}
If make_table: {"title": "...", "description": "...", "headers": ["..."], "rows": [["..."], ...]}
Incorporate the suggested fix while preserving existing factual context. Label mock data as illustrative if relevant."""

SUMMARY_SYSTEM_PROMPT = """Write a concise executive summary (2 to 3 sentences) of the final document.
Return plain text only, without formatting or JSON."""


class AgentOrchestrator:
    """Coordinates planning, execution, review, and document generation."""

    def __init__(
        self,
        llm_client: LLMClient,
        run_store: RunStore,
        output_dir: str | None = None,
    ) -> None:
        self.llm = llm_client
        self.store = run_store
        self.output_dir = output_dir or os.getenv("DOCX_OUTPUT_DIR", "outputs")

    async def run(self, user_request: str) -> AgentResponse:
        """Execute the full agent workflow for a business request."""
        run_id = uuid.uuid4().hex[:12]
        providers_used: set[str] = set()

        logger.info("Starting planner for run %s", run_id)
        plan, plan_provider = await self._create_plan(user_request)
        providers_used.add(plan_provider)

        logger.info("Executing %d tasks for run %s", len(plan.tasks), run_id)
        task_results, exec_providers = await self._execute_plan(user_request, plan)
        providers_used.update(exec_providers)

        logger.info("Reviewing assembled document for run %s", run_id)
        review, final_results, rev_providers = await self._review_and_revise(
            user_request, plan, task_results
        )
        providers_used.update(rev_providers)

        summary_text, sum_provider = await self._generate_summary(
            plan.title, plan.doc_type, final_results
        )
        providers_used.add(sum_provider)

        docx_path = build_docx(
            run_id=run_id,
            title=plan.title,
            doc_type=plan.doc_type,
            assumptions=plan.assumptions,
            tasks=final_results,
            output_dir=self.output_dir,
        )

        record = RunRecord(
            run_id=run_id,
            request=user_request,
            doc_type=plan.doc_type,
            title=plan.title,
            summary=summary_text,
            assumptions=plan.assumptions,
            tasks=final_results,
            review=review,
            providers_used=sorted(providers_used),
            docx_path=docx_path,
        )
        self.store.save(record)

        task_summaries = [
            TaskSummary(id=t.task_id, title=t.title, status=t.status)
            for t in final_results
        ]
        all_done = all(t.status == "done" for t in final_results)

        return AgentResponse(
            run_id=run_id,
            doc_type=plan.doc_type,
            title=plan.title,
            summary=summary_text,
            assumptions=plan.assumptions,
            tasks=task_summaries,
            review=review,
            providers_used=sorted(providers_used),
            download_url=f"/download/{run_id}",
            complete=all_done,
        )

    async def _create_plan(self, user_request: str) -> tuple[PlannerOutput, str]:
        user_prompt = f"User Request: {user_request}\n\nProduce the document plan JSON."
        last_error: Exception | None = None

        for attempt in range(2):
            prompt = (
                user_prompt
                if attempt == 0
                else f"{user_prompt}\n\nPrevious attempt failed schema validation: {last_error}. Ensure between 5 and 9 tasks are returned."
            )
            response = await self.llm.generate(PLANNER_SYSTEM_PROMPT, prompt)
            try:
                data = extract_json(response.text)
                plan = PlannerOutput.model_validate(data)
                self._ensure_table_task(plan)
                return plan, response.provider
            except (ValueError, ValidationError, LLMProviderError) as exc:
                last_error = exc
                logger.warning("Planner attempt %d failed: %s", attempt + 1, exc)

        raise PlanningError(f"Planning failed after validation retries: {last_error}")

    def _ensure_table_task(self, plan: PlannerOutput) -> None:
        """Convert a candidate section to make_table if a plan/proposal lacks tables."""
        if plan.doc_type in ("project_plan", "proposal") and not any(
            t.tool == "make_table" for t in plan.tasks
        ):
            keywords = ("timeline", "schedule", "budget", "risk")
            for t in plan.tasks:
                title_lower = t.title.lower()
                if any(k in title_lower for k in keywords):
                    logger.warning(
                        "Converting task %d ('%s') to make_table for %s document",
                        t.id,
                        t.title,
                        plan.doc_type,
                    )
                    t.tool = "make_table"
                    break

    def _append_rolling_context(
        self,
        rolling_context: list[str],
        task: PlanTask,
        content: SectionContent | TableContent | None,
    ) -> None:
        if isinstance(content, SectionContent):
            snippet = content.paragraphs[0][:120] if content.paragraphs else task.title
            rolling_context.append(f"{task.title}: {snippet}")
        elif isinstance(content, TableContent):
            rolling_context.append(f"{task.title}: table with {len(content.rows)} rows")

    async def _run_task_attempt(
        self,
        task: PlanTask,
        result: TaskResult,
        user_request: str,
        assumptions: list[str],
        context_summary: str,
        tool_fn: Any,
    ) -> tuple[bool, str | None]:
        result.status = "running"
        try:
            content, provider = await tool_fn(
                self.llm,
                task,
                user_request,
                assumptions,
                context_summary,
            )
            result.content = content
            result.status = "done"
            result.error_message = None
            return True, provider
        except (ValueError, ValidationError, LLMProviderError, RuntimeError) as exc:
            logger.warning("Task %d (%s) failed attempt: %s", task.id, task.title, exc)
            result.error_message = str(exc)
            return False, None

    async def _execute_single_task(
        self,
        task: PlanTask,
        result: TaskResult,
        user_request: str,
        assumptions: list[str],
        context_summary: str,
        tool_fn: Any,
    ) -> tuple[bool, str | None]:
        success, provider = await self._run_task_attempt(
            task, result, user_request, assumptions, context_summary, tool_fn
        )
        if success:
            return True, provider

        # If a task fails after its own retries, wait 5s and try once more
        logger.info(
            "Task %d (%s) failed; waiting 5s before task-level retry",
            task.id,
            task.title,
        )
        await asyncio.sleep(5)
        success, provider = await self._run_task_attempt(
            task, result, user_request, assumptions, context_summary, tool_fn
        )
        if success:
            return True, provider

        result.status = "failed"
        return False, None

    async def _execute_plan(
        self, user_request: str, plan: PlannerOutput
    ) -> tuple[list[TaskResult], set[str]]:
        results: list[TaskResult] = [
            TaskResult(
                task_id=task.id,
                title=task.title,
                tool=task.tool,
                status="pending",
            )
            for task in plan.tasks
        ]
        providers_used: set[str] = set()
        rolling_context: list[str] = []
        tool_map = {
            "write_section": write_section,
            "make_table": make_table,
        }

        for task, result in zip(plan.tasks, results):
            context_summary = " ".join(rolling_context[-3:])
            tool_fn = tool_map.get(task.tool, write_section)
            success, provider = await self._execute_single_task(
                task,
                result,
                user_request,
                plan.assumptions,
                context_summary,
                tool_fn,
            )
            if success and provider:
                providers_used.add(provider)
                self._append_rolling_context(rolling_context, task, result.content)

        # After all tasks finish, if any failed, run one extra pass over just the failed tasks
        failed_tasks = [
            (task, result)
            for task, result in zip(plan.tasks, results)
            if result.status == "failed"
        ]
        if failed_tasks:
            logger.info("Running recovery pass for %d failed task(s)", len(failed_tasks))
            for task, result in failed_tasks:
                context_summary = " ".join(rolling_context[-3:])
                tool_fn = tool_map.get(task.tool, write_section)
                success, provider = await self._execute_single_task(
                    task,
                    result,
                    user_request,
                    plan.assumptions,
                    context_summary,
                    tool_fn,
                )
                if success and provider:
                    providers_used.add(provider)
                    self._append_rolling_context(rolling_context, task, result.content)

        return results, providers_used

    async def _review_and_revise(
        self,
        user_request: str,
        plan: PlannerOutput,
        tasks: list[TaskResult],
    ) -> tuple[ReviewOutput, list[TaskResult], set[str]]:
        providers_used: set[str] = set()

        doc_dump = []
        for t in tasks:
            if t.status == "failed" or t.content is None:
                doc_dump.append(f"Section {t.task_id}: {t.title} [FAILED TO GENERATE]")
            elif isinstance(t.content, SectionContent):
                p_text = " ".join(t.content.paragraphs)
                doc_dump.append(f"Section {t.task_id}: {t.content.title}\n{p_text}")
            elif isinstance(t.content, TableContent):
                doc_dump.append(
                    f"Section {t.task_id}: {t.content.title} (Table with headers: {t.content.headers})"
                )

        review_prompt = (
            f"Original Request: {user_request}\n"
            f"Doc Type: {plan.doc_type}\n"
            f"Assumptions:\n" + "\n".join(f"- {a}" for a in plan.assumptions) + "\n\n"
            "Generated Sections:\n" + "\n\n".join(doc_dump) + "\n\n"
            "Assess the document and return the review JSON."
        )

        try:
            resp = await self.llm.generate(REVIEWER_SYSTEM_PROMPT, review_prompt)
            providers_used.add(resp.provider)
            review_data = extract_json(resp.text)
            review = ReviewOutput.model_validate(review_data)
        except (ValueError, ValidationError, LLMProviderError, RuntimeError) as exc:
            logger.warning("Review step encountered an issue: %s. Skipping review.", exc)
            review = ReviewOutput(
                summary="Review skipped because the reviewer call failed.",
                needs_fix=False,
                issues=[],
            )

        if not review.needs_fix or not review.issues:
            return review, tasks, providers_used

        task_map = {t.task_id: t for t in tasks}
        for issue in review.issues:
            target_task = task_map.get(issue.section_id)
            if not target_task or target_task.status != "done" or target_task.content is None:
                continue

            current_content_json = target_task.content.model_dump_json()
            revise_prompt = (
                f"Business Request: {user_request}\n"
                f"Section: {target_task.title}\n"
                f"Tool: {target_task.tool}\n"
                f"Current Content: {current_content_json}\n"
                f"Issue: {issue.issue}\n"
                f"Suggested Fix: {issue.suggested_fix}\n\n"
                f"Revise and return updated JSON."
            )
            try:
                rev_resp = await self.llm.generate(REVISE_SYSTEM_PROMPT, revise_prompt)
                providers_used.add(rev_resp.provider)
                rev_data = extract_json(rev_resp.text)
                if target_task.tool == "write_section":
                    target_task.content = SectionContent.model_validate(rev_data)
                else:
                    target_task.content = TableContent.model_validate(rev_data)
                logger.info("Applied revision to section %d", issue.section_id)
            except (ValueError, ValidationError, LLMProviderError, RuntimeError) as exc:
                logger.warning(
                    "Revision failed for section %d: %s. Preserving current version.",
                    issue.section_id,
                    exc,
                )

        return review, tasks, providers_used

    async def _generate_summary(
        self, title: str, doc_type: str, tasks: list[TaskResult]
    ) -> tuple[str, str]:
        completed = [t.title for t in tasks if t.status == "done"]
        sections_overview = ", ".join(completed)
        prompt = (
            f"Document Title: {title}\n"
            f"Document Type: {doc_type}\n"
            f"Completed Sections: {sections_overview}\n\n"
            f"Write a 2 to 3 sentence executive summary for this document."
        )
        try:
            resp = await self.llm.generate(SUMMARY_SYSTEM_PROMPT, prompt)
            return resp.text.strip(), resp.provider
        except (LLMProviderError, RuntimeError):
            n = len(completed)
            m = len(tasks)
            completed_str = f" ({', '.join(completed)})" if completed else ""
            return f"{title}: {n} of {m} sections generated{completed_str}.", "system"
