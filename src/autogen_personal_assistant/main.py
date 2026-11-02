"""Run the Magentic-One personal assistant in single-query or interactive mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
import warnings
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path

# Support both direct execution and imports without shadowing the original main.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import MagenticOneGroupChat
from dotenv import load_dotenv

from autogen_personal_assistant.team import (
    create_model_client,
    create_team,
    load_existing_tools,
)
from autogen_personal_assistant.tracing import TaskTrace, TraceRecorder


def configure_runtime_warnings() -> None:
    """Hide the known harmless SDK warning produced for Magentic-One ledgers.

    AutoGen 0.7.5 asks the OpenAI SDK to parse the coordinator's ledger into a
    Pydantic model.  The SDK logs that parsed model with a field typed as
    ``None``, which produces this serializer warning even though AutoGen has
    already consumed the ledger successfully.  Limit the filter to that exact
    internal field so unrelated Pydantic warnings remain visible.
    """
    warnings.filterwarnings(
        "ignore",
        # Pydantic indents the detailed item in its formatted warning.
        message=r"Pydantic serializer warnings:\s+PydanticSerializationUnexpectedValue\(Expected `none`.*field_name='parsed'",
        category=UserWarning,
    )


class PersonalAssistantSession:
    """Keep workspace changes and conversation history for one interactive session.

    Attributes:
        workspace: Isolated copy used by the existing MCP servers.
        history: Completed user/assistant turns supplied as context to later requests.
        team: Magentic-One team, reset before each new request.
    """

    def __init__(self, workspace: Path, date: str, team: MagenticOneGroupChat) -> None:
        """Initialize session state.

        Args:
            workspace: Session workspace directory.
            date: Authoritative workspace date.
            team: Configured MagenticOneGroupChat instance.
        """
        self.workspace = workspace
        self.date = date
        self.team = team
        self.history: list[dict[str, str]] = []

    async def run(self, query: str, trace: TaskTrace | None = None, verbose: bool = False) -> str:
        """Process a request with fresh team ledgers and explicit session context.

        Args:
            query: Current user request.
            trace: Optional recorder for actual tool events.
            verbose: Whether to print intermediate messages to stderr.

        Returns:
            The manager's final answer.

        Raises:
            RuntimeError: If execution ends without a final answer or reaches its limit.
        """
        await self.team.reset()
        task = (
            f"Workspace date: {self.date}. This overrides stale dates in tool descriptions.\n"
            "Use tool evidence for workspace facts. The available email, calendar, task, "
            "contact, and note tools search only the local workspace; do not ask for "
            "external-account credentials or access. Perform only the current request.\n"
            "Earlier turns are context, not actions to repeat:\n"
            f"{json.dumps(self.history, ensure_ascii=False)}\n\n"
            f"Current user request:\n{query}"
        )
        result = None
        async for event in self.team.run_stream(task=task):
            if isinstance(event, TaskResult):
                result = event
            else:
                if trace is not None:
                    trace.observe(event)
                if verbose:
                    print(f"[{event.source}] {event.to_text()}", file=sys.stderr, flush=True)
        if result is None:
            raise RuntimeError("Team did not return a task result.")
        if result.stop_reason == "Max rounds reached.":
            raise RuntimeError("Task incomplete: the team reached its 20-turn limit. Workspace changes may exist.")
        if not result.messages or not isinstance(result.messages[-1], TextMessage):
            raise RuntimeError("Team did not return a final text answer.")
        answer = result.messages[-1].content
        self.history.append({"user": query, "assistant": answer})
        if trace is not None:
            trace.root.set_outputs({"answer": answer, "stop_reason": result.stop_reason})
        return answer


def parse_args() -> argparse.Namespace:
    """Parse the CLI options shared with the original assistant.

    Returns:
        Validated command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Run one request and exit.")
    parser.add_argument("--session", action="store_true", help="Start an interactive session.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--trace_dir", type=Path, default=Path("traces"))
    parser.add_argument("--mlflow_db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--exp_name", default="pa_magentic_one")
    parser.add_argument("--tracking_uri", default="")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds allowed per request.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


async def execute_query(session: PersonalAssistantSession, query: str, args: argparse.Namespace,
                        recorder: TraceRecorder | None) -> bool:
    """Run a bounded request and print its answer or an explicit failure.

    Args:
        session: Current workspace and conversation state.
        query: User request.
        args: CLI timeout and verbosity settings.
        recorder: Optional MLflow trace exporter.

    Returns:
        Whether execution completed normally.
    """
    try:
        with recorder.task(query, session.workspace) if recorder else nullcontext() as trace:
            answer = await asyncio.wait_for(session.run(query, trace, args.verbose), args.timeout)
        print(answer, flush=True)
        return True
    except asyncio.TimeoutError:
        print("Task incomplete: request timed out. Workspace changes may exist.", file=sys.stderr)
    except Exception as exc:
        print(f"Task failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return False


async def main() -> int:
    """Start a session, execute requests, and close all external clients.

    Returns:
        Zero on success, or one when a request or setup fails.
    """
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    configure_runtime_warnings()
    load_dotenv(REPO_ROOT / ".env", override=True)
    try:
        async with AsyncExitStack() as stack:
            model = create_model_client()
            stack.push_async_callback(model.close)
            workspace_root = REPO_ROOT / "tmp_workspaces"
            workspace_root.mkdir(exist_ok=True)
            workspace = Path(tempfile.mkdtemp(prefix="session_", dir=workspace_root))
            shutil.copytree(REPO_ROOT / "workspace", workspace, dirs_exist_ok=True)
            date = json.loads((REPO_ROOT / "user_tasks.json").read_text())["workspace_date"]
            tools = await load_existing_tools(workspace, stack)
            session = PersonalAssistantSession(workspace, date, create_team(model, tools, date))
            recorder = TraceRecorder(
                REPO_ROOT / args.trace_dir, REPO_ROOT / args.mlflow_db, args.exp_name, args.tracking_uri,
            ) if args.mlflow else None
            if args.query is not None and not args.session:
                return 0 if await execute_query(session, args.query, args, recorder) else 1
            print(f"Personal assistant session started.\nSession workspace: {workspace}")
            print("Type a task, or 'exit' / 'quit' to end the session.")
            while True:
                try:
                    query = input("pa> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if query.lower() in {"exit", "quit"}:
                    break
                if query:
                    await execute_query(session, query, args, recorder)
            return 0
    except Exception as exc:
        print(f"Session failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
