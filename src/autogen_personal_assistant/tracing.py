"""Record one MLflow trace per request without changing any existing tool."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow
from autogen_agentchat.messages import ToolCallExecutionEvent, ToolCallRequestEvent
from mlflow.entities import LiveSpan
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING


class TaskTrace:
    """Correlate streamed tool requests and results inside one task trace.

    Attributes:
        root: Parent span for all observed workspace tool calls.
        pending: Tool spans awaiting execution results, indexed by call ID.
    """

    def __init__(self, root: LiveSpan) -> None:
        """Initialize the recorder.

        Args:
            root: Root span of the current user request.
        """
        self.root = root
        self.pending: dict[str, LiveSpan] = {}

    def observe(self, event: object) -> None:
        """Record actual tool events emitted by AutoGen.

        Args:
            event: A message yielded by the team's stream.
        """
        if isinstance(event, ToolCallRequestEvent):
            for call in event.content:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    arguments = {"invalid_arguments": call.arguments}
                self.pending[call.id] = mlflow.start_span_no_context(
                    call.name, span_type="TOOL", parent_span=self.root,
                    inputs=arguments, attributes={"agent": event.source, "call_id": call.id},
                )
        elif isinstance(event, ToolCallExecutionEvent):
            for result in event.content:
                span = self.pending.pop(result.call_id, None)
                if span is not None:
                    span.set_outputs({"content": result.content, "is_error": result.is_error})
                    span.end(status="ERROR" if result.is_error else "OK")

    def close_pending(self) -> None:
        """Mark tool calls without execution results as interrupted."""
        for span in self.pending.values():
            span.set_outputs({"error": "Execution interrupted before a tool result was received."})
            span.end(status="ERROR")
        self.pending.clear()


class TraceRecorder:
    """Configure MLflow and export exactly the current request's trace."""

    def __init__(self, directory: Path, database: Path, experiment: str, tracking_uri: str = "") -> None:
        """Configure tracking after AutoGen model and agent classes are imported.

        Args:
            directory: Destination for exported trace JSON files.
            database: SQLite tracking database used when no URI is supplied.
            experiment: MLflow experiment name.
            tracking_uri: Optional external tracking location.
        """
        import mlflow.autogen

        # Export synchronously so the completed root is immediately retrievable.
        MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.set(False)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        if not tracking_uri:
            database.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{database.resolve()}")
        mlflow.set_experiment(experiment)
        mlflow.autogen.autolog(log_traces=True, silent=True)

    @contextmanager
    def task(self, query: str, workspace: Path) -> Iterator[TaskTrace]:
        """Enclose execution and export its complete trace, including failures.

        Args:
            query: Current user request, without previous conversation turns.
            workspace: Active session workspace.

        Yields:
            Recorder for streamed tool evidence.

        Raises:
            RuntimeError: If the completed trace cannot be retrieved.
        """
        with mlflow.start_run():
            try:
                with mlflow.start_span("personal_assistant_task", span_type="CHAIN") as root:
                    root.set_inputs({"query": query, "workspace": str(workspace)})
                    trace = TaskTrace(root)
                    try:
                        yield trace
                    finally:
                        trace.close_pending()
            finally:
                completed = mlflow.get_trace(root.trace_id)
                if completed is None:
                    raise RuntimeError(f"Could not retrieve task trace {root.trace_id}")
                path = self.directory / f"{root.trace_id}.json"
                path.write_text(completed.to_json(pretty=True), encoding="utf-8")
                print(f"Trace JSON exported: {path}", flush=True)
