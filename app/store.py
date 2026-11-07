import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.schemas import ReviewOutput, TaskResult


@dataclass
class RunRecord:
    """Represents a persisted execution run."""

    run_id: str
    request: str
    doc_type: str
    title: str
    summary: str
    assumptions: list[str]
    tasks: list[TaskResult]
    review: ReviewOutput
    providers_used: list[str]
    docx_path: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class RunStore:
    """Thread-safe in-memory store for agent execution records."""

    def __init__(self) -> None:
        self._store: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: RunRecord) -> None:
        """Save or overwrite a run record."""
        with self._lock:
            self._store[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by its identifier."""
        with self._lock:
            return self._store.get(run_id)

    def count(self) -> int:
        """Return the total number of stored records."""
        with self._lock:
            return len(self._store)
