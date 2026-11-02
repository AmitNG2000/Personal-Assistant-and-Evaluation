"""Evaluate Magentic-One using Task 2's checks and read-only evidence tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_assistant import (
    CHECKS,
    Evaluator,
    Evaluator_Agent,
    write_reports,
)


class AutoGenEvaluator(Evaluator):
    """Launch Task 3 while inheriting Task 2's evidence checks unchanged."""

    def __init__(self, task: dict, workspace_date: str, output_dir: Path,
                 timeout: int = 300, verbose: bool = False) -> None:
        """Configure one evaluation case.

        Args:
            task: Unmodified task definition from user_tasks.json.
            workspace_date: Date shared with the original assistant.
            output_dir: Task 3 report and trace directory.
            timeout: Maximum seconds for each request and judge review.
            verbose: Whether to display the assistant's captured output.
        """
        super().__init__(task, workspace_date, timeout, verbose)
        self.output_dir = output_dir

    @staticmethod
    def review_allows_overall_pass(review: dict) -> bool:
        """Return whether a tri-state evaluator review is non-blocking."""
        return review.get("verdict") in {"PASS", "UNKNOWN"}

    @staticmethod
    def unavailable_review(error: Exception) -> dict:
        """Represent an evaluator transport or parsing error as an unknown review."""
        return {
            "verdict": "UNKNOWN",
            "reason": f"Evaluator review unavailable: {type(error).__name__}: {error}",
            "evidence": [],
        }

    def run(self) -> dict:
        """Collect execution evidence and apply the original evaluation criteria.

        Returns:
            Task 2-compatible result, augmented with trace location and count.
        """
        try:
            before = self.snapshot(REPO_ROOT / "workspace")
            sessions = set((REPO_ROOT / "tmp_workspaces").glob("session_*"))
            trace_root = self.output_dir / "mlflow"
            trace_root.mkdir(parents=True, exist_ok=True)
            traces = Path(tempfile.mkdtemp(prefix=f"task_{self.task['id']}_", dir=trace_root))
            self.result["trace_directory"] = str(traces)
            command = [
                sys.executable, str(REPO_ROOT / "tasks/task_3_agentic_system_implementation/main.py"),
                "--query", self.task["query"], "--mlflow", "--trace_dir", str(traces),
                "--mlflow_db", str(self.output_dir / "mlflow.db"),
                "--timeout", str(self.timeout),
            ]
            # Give the assistant time to export a failed trace and close MCP sessions.
            with subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=(os.name == "posix"),
            ) as process:
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout + 30)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.communicate()
                    raise RuntimeError("Assistant exceeded the request timeout and cleanup allowance.")
            if self.verbose:
                print(stdout, stderr, flush=True)
            self.result["answer"] = "\n".join(
                line for line in stdout.splitlines() if not line.startswith("Trace JSON exported:")
            ).strip()
            self.result["actual_tool_calls"] = self.read_tool_calls(traces)
            self.result["trace_count"] = len(list(traces.glob("*.json")))
            if self.result["trace_count"] != 1:
                self.result["failures"].append("Expected exactly one exported trace for the request.")
            created = set((REPO_ROOT / "tmp_workspaces").glob("session_*")) - sessions
            if len(created) != 1:
                raise RuntimeError("Cannot identify a unique workspace; run evaluations sequentially.")
            workspace = created.pop()
            self.result["session_workspace"] = str(workspace)
            after = self.snapshot(workspace)
            self.check_expectations(before, after)
            if process.returncode:
                raise RuntimeError(stderr.strip() or f"Assistant exited with {process.returncode}")
            try:
                review = asyncio.run(asyncio.wait_for(
                    Evaluator_Agent(workspace).evaluate(self.result, before, after), self.timeout,
                ))
            except Exception as exc:
                # The assistant and deterministic checks completed. A judge transport
                # or parsing failure is inconclusive, not evidence that the task failed.
                review = self.unavailable_review(exc)
            self.result["agent_review"] = review
            self.result["agent_pass"] = self.review_allows_overall_pass(review)
            if review["verdict"] == "FAIL":
                self.result["failures"].append(f"AI review (FAIL): {review['reason']}")
            elif review["verdict"] == "UNKNOWN":
                self.result["failures"].append(
                    f"AI review (UNKNOWN, non-blocking): {review['reason']}"
                )
            self.result["overall_pass"] = all(self.result[key] for key in CHECKS)
        except Exception as exc:
            self.result["error"] = f"{type(exc).__name__}: {exc}"
        return self.result


def main() -> int:
    """Evaluate tasks sequentially and preserve reports after every case.

    Returns:
        Zero when all cases and their single-trace checks pass, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, default=REPO_ROOT / "user_tasks.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.tasks_file.read_text(encoding="utf-8"))
    date = json.loads((REPO_ROOT / "user_tasks.json").read_text(encoding="utf-8"))["workspace_date"]
    if args.timeout <= 0 or not payload.get("tasks"):
        parser.error("Provide a positive timeout and a nonempty task list.")
    if payload.get("workspace_date", date) != date:
        parser.error("workspace_date must match the repository's user_tasks.json.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for task in payload["tasks"]:
        print(f"Running and reviewing task {task['id']}...", flush=True)
        result = AutoGenEvaluator(task, date, output, args.timeout, args.verbose).run()
        results.append(result)
        write_reports(results, output)
        (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Task {task['id']}: {'PASS' if result['overall_pass'] else 'FAIL'}", flush=True)
    print(f"Reports: {output / 'user_tasks_results.md'} and user_tasks_results.csv")
    return 0 if all(item["overall_pass"] and item.get("trace_count") == 1 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
