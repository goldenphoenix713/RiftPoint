"""Multiverse speculative runner for concurrent branch execution in LangGraph."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from riftpoint.logger import logger
from riftpoint.runner.branch import BranchManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

    from riftpoint.checkpointer.base import BaseRiftSaver


@dataclass
class BranchSpec:
    """Specification for a candidate speculative timeline."""

    name: str
    input_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchResult:
    """Outcome and state trajectory from a candidate branch execution."""

    branch_name: str
    output: dict[str, Any] | None = None
    final_config: RunnableConfig | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Return True if branch executed without error, False otherwise."""
        return self.error is None and self.output is not None


class RiftRunner:
    """Orchestrates speculative branch execution over multiversal timelines."""

    def __init__(
        self,
        saver: BaseRiftSaver,
        branch_manager: BranchManager | None = None,
    ) -> None:
        """Initialize the speculative RiftRunner.

        Args:
            saver: The checkpointer backend managing timeline state.
            branch_manager: Optional branch manager instance (defaults to new instance).
        """
        self.saver = saver
        self.branch_manager = branch_manager or BranchManager(saver)

    def _execute_branch_sync(
        self,
        graph: Pregel[Any, Any],
        thread_id: str,
        from_checkpoint: str | None,
        spec: BranchSpec,
    ) -> BranchResult:
        """Execute a single candidate branch synchronously."""
        branch_config = self.branch_manager.create_branch(
            thread_id=thread_id,
            branch_name=spec.name,
            from_checkpoint=from_checkpoint,
        )
        input_payload = spec.input_data or {}
        try:
            output = graph.invoke(input_payload, config=branch_config)
            return BranchResult(
                branch_name=spec.name,
                output=output,
                final_config=branch_config,
                metadata=spec.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Branch '%s' execution failed: %s", spec.name, exc)
            return BranchResult(
                branch_name=spec.name,
                error=exc,
                final_config=branch_config,
                metadata=spec.metadata,
            )

    def run_parallel_branches(
        self,
        graph: Pregel[Any, Any],
        initial_config: RunnableConfig,
        branch_specs: Sequence[BranchSpec],
        *,
        max_workers: int | None = None,
    ) -> dict[str, BranchResult]:
        """Concurrently execute speculative candidate branches synchronously.

        Args:
            graph: Compiled LangGraph Pregel state graph.
            initial_config: Root configuration with thread_id and parent
                checkpoint.
            branch_specs: Specifications for each candidate branch to explore.
            max_workers: Optional thread pool concurrency limit.

        Returns:
            Dictionary mapping branch names to their execution results.
        """
        thread_id = initial_config["configurable"]["thread_id"]
        from_checkpoint = initial_config["configurable"].get("checkpoint_id")

        results: dict[str, BranchResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_branch_sync,
                    graph,
                    thread_id,
                    from_checkpoint,
                    spec,
                ): spec.name
                for spec in branch_specs
            }
            for future, branch_name in futures.items():
                try:
                    results[branch_name] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[branch_name] = BranchResult(
                        branch_name=branch_name,
                        error=exc,
                    )
        return results

    async def _execute_branch_async(
        self,
        graph: Pregel[Any, Any],
        thread_id: str,
        from_checkpoint: str | None,
        spec: BranchSpec,
    ) -> BranchResult:
        """Execute a single candidate branch asynchronously."""
        branch_config = self.branch_manager.create_branch(
            thread_id=thread_id,
            branch_name=spec.name,
            from_checkpoint=from_checkpoint,
        )
        input_payload = spec.input_data or {}
        try:
            output = await graph.ainvoke(input_payload, config=branch_config)
            return BranchResult(
                branch_name=spec.name,
                output=output,
                final_config=branch_config,
                metadata=spec.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Async branch '%s' execution failed: %s", spec.name, exc)
            return BranchResult(
                branch_name=spec.name,
                error=exc,
                final_config=branch_config,
                metadata=spec.metadata,
            )

    async def arun_parallel_branches(
        self,
        graph: Pregel[Any, Any],
        initial_config: RunnableConfig,
        branch_specs: Sequence[BranchSpec],
    ) -> dict[str, BranchResult]:
        """Concurrently execute speculative candidate branches asynchronously.

        Args:
            graph: Compiled LangGraph Pregel state graph.
            initial_config: Root configuration with thread_id and parent
                checkpoint.
            branch_specs: Specifications for each candidate branch to explore.

        Returns:
            Dictionary mapping branch names to their execution results.
        """
        thread_id = initial_config["configurable"]["thread_id"]
        from_checkpoint = initial_config["configurable"].get("checkpoint_id")

        tasks = [
            self._execute_branch_async(graph, thread_id, from_checkpoint, spec)
            for spec in branch_specs
        ]
        results_list = await asyncio.gather(*tasks)
        return {res.branch_name: res for res in results_list}
