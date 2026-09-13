"""Declarative decorator for speculative node execution in LangGraph."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from riftpoint.logger import logger
from riftpoint.runner.multiverse import BranchResult

if TYPE_CHECKING:
    from riftpoint.collapse.evaluators import BaseEvaluator


@dataclass
class SpeculativeNodeConfig:
    """Configuration for a speculative node race.

    Attributes:
        tools: Candidate tool functions, each taking state dict and
            returning a state update dict.
        evaluator: Evaluator instance for scoring tool outputs.
        prune_discarded: Whether to discard losing tool results from
            metadata.
        max_workers: Thread pool size for sync racing. None uses the
            default ThreadPoolExecutor sizing.
        fallback_on_all_fail: If all tools fail, call the original
            decorated function as fallback.
    """

    tools: list[Callable[..., dict[str, Any]]]
    evaluator: BaseEvaluator
    prune_discarded: bool = True
    max_workers: int | None = None
    fallback_on_all_fail: bool = True


@dataclass
class SpeculativeRaceMeta:
    """Metadata from a speculative tool race within a node.

    Attributes:
        winner_tool: Name of the winning tool function.
        scores: Map of tool name to evaluation score.
        all_succeeded: Whether all candidate tools executed without error.
        used_fallback: Whether the original function was used as fallback.
    """

    winner_tool: str
    scores: dict[str, float] = field(default_factory=dict)
    all_succeeded: bool = True
    used_fallback: bool = False


def _get_tool_name(tool: Callable[..., Any]) -> str:
    """Extract a human-readable name from a tool callable."""
    return getattr(tool, "__name__", repr(tool))


def _run_tool_safe(
    tool: Callable[..., dict[str, Any]],
    state: dict[str, Any],
) -> BranchResult:
    """Execute a single tool safely, wrapping output as a BranchResult."""
    tool_name = _get_tool_name(tool)
    try:
        output = tool(state)
        return BranchResult(
            branch_name=tool_name,
            output=output,
            metadata={"tool": tool_name},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Speculative tool '%s' failed: %s",
            tool_name,
            exc,
        )
        return BranchResult(
            branch_name=tool_name,
            error=exc,
            metadata={"tool": tool_name},
        )


async def _arun_tool_safe(
    tool: Callable[..., Any],
    state: dict[str, Any],
) -> BranchResult:
    """Execute a single tool safely in async context."""
    tool_name = _get_tool_name(tool)
    try:
        if inspect.iscoroutinefunction(tool):
            output = await tool(state)
        else:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(None, tool, state)
        return BranchResult(
            branch_name=tool_name,
            output=output,
            metadata={"tool": tool_name},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Async speculative tool '%s' failed: %s",
            tool_name,
            exc,
        )
        return BranchResult(
            branch_name=tool_name,
            error=exc,
            metadata={"tool": tool_name},
        )


def _select_winner(
    results: list[BranchResult],
    evaluator: BaseEvaluator,
) -> tuple[BranchResult, dict[str, float]]:
    """Evaluate and select the winning tool result.

    Args:
        results: List of BranchResults from tool executions.
        evaluator: Evaluator for scoring.

    Returns:
        Tuple of (winning BranchResult, scores dict).
    """
    scores: dict[str, float] = {}
    for res in results:
        eval_result = evaluator.evaluate(res)
        scores[res.branch_name] = eval_result.score

    ranked = sorted(
        results,
        key=lambda r: scores.get(r.branch_name, 0.0),
        reverse=True,
    )
    return ranked[0], scores


async def _aselect_winner(
    results: list[BranchResult],
    evaluator: BaseEvaluator,
) -> tuple[BranchResult, dict[str, float]]:
    """Asynchronously evaluate and select the winning tool result.

    Args:
        results: List of BranchResults from tool executions.
        evaluator: Evaluator for scoring.

    Returns:
        Tuple of (winning BranchResult, scores dict).
    """
    tasks = [evaluator.aevaluate(res) for res in results]
    eval_results = await asyncio.gather(*tasks)
    scores: dict[str, float] = {
        res.branch_name: er.score for res, er in zip(results, eval_results, strict=True)
    }

    ranked = sorted(
        results,
        key=lambda r: scores.get(r.branch_name, 0.0),
        reverse=True,
    )
    return ranked[0], scores


def _build_race_meta(
    winner: BranchResult,
    scores: dict[str, float],
    results: list[BranchResult],
    *,
    used_fallback: bool = False,
) -> SpeculativeRaceMeta:
    """Build structured metadata from a completed race."""
    all_succeeded = all(r.is_success for r in results)
    return SpeculativeRaceMeta(
        winner_tool=winner.branch_name,
        scores=scores,
        all_succeeded=all_succeeded,
        used_fallback=used_fallback,
    )


F = TypeVar("F", bound=Callable[..., Any])


def speculative_node(
    tools: Sequence[Callable[..., Any]],
    evaluator: BaseEvaluator,
    *,
    prune_discarded: bool = True,
    max_workers: int | None = None,
    fallback_on_all_fail: bool = True,
) -> Callable[[F], F]:
    """Decorator that turns a LangGraph node into a speculative tool race.

    Races all candidate tool functions concurrently on the node's input
    state, evaluates their outputs, and returns the winning tool's state
    update. If all tools fail and ``fallback_on_all_fail`` is True, the
    original decorated function is called as a safe fallback.

    Args:
        tools: Candidate tool functions, each ``(state) -> dict``.
        evaluator: Evaluator instance for scoring tool outputs.
        prune_discarded: Whether to discard losing results from metadata.
        max_workers: Thread pool size for synchronous tool racing.
        fallback_on_all_fail: Use the original function as fallback if
            all tools fail.

    Returns:
        Decorator that wraps a LangGraph node function.

    Example:
        >>> @speculative_node(
        ...     tools=[fetch_cache, fetch_rag, fetch_web],
        ...     evaluator=HeuristicEvaluator(scorer=my_scorer),
        ... )
        ... def research(state: AgentState) -> dict:
        ...     return {"answer": "fallback"}
    """
    tool_list = list(tools)
    _ = prune_discarded

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            return cast(
                "F",
                _wrap_async(
                    fn,
                    tool_list,
                    evaluator,
                    max_workers,
                    fallback_on_all_fail=fallback_on_all_fail,
                ),
            )
        return cast(
            "F",
            _wrap_sync(
                fn,
                tool_list,
                evaluator,
                max_workers,
                fallback_on_all_fail=fallback_on_all_fail,
            ),
        )

    return decorator


def _wrap_sync(
    fn: Callable[..., dict[str, Any]],
    tools: list[Callable[..., dict[str, Any]]],
    evaluator: BaseEvaluator,
    max_workers: int | None,
    *,
    fallback_on_all_fail: bool,
) -> Callable[..., dict[str, Any]]:
    """Create synchronous speculative wrapper."""

    @functools.wraps(fn)
    def wrapper(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        results = _race_tools_sync(tools, state, max_workers)
        successful = [r for r in results if r.is_success]

        if not successful:
            _, scores = _select_winner(results, evaluator)
            if fallback_on_all_fail:
                logger.warning(
                    "All %d speculative tools failed for '%s'. Using fallback.",
                    len(results),
                    fn.__name__,
                )
                fallback_output = fn(state, **kwargs)
                meta = _build_race_meta(results[0], scores, results, used_fallback=True)
                return {**fallback_output, "__speculative_meta__": meta}
            meta = _build_race_meta(results[0], scores, results)
            return {"__speculative_meta__": meta}

        winner, scores = _select_winner(results, evaluator)
        if not winner.is_success and successful:
            winner = successful[0]

        meta = _build_race_meta(winner, scores, results)
        output = winner.output or {}
        logger.info(
            "Speculative node '%s': winner='%s' (score=%.3f)",
            fn.__name__,
            winner.branch_name,
            scores.get(winner.branch_name, 0.0),
        )
        return {**output, "__speculative_meta__": meta}

    return wrapper


def _wrap_async(
    fn: Callable[..., Any],
    tools: list[Callable[..., dict[str, Any]]],
    evaluator: BaseEvaluator,
    max_workers: int | None,
    *,
    fallback_on_all_fail: bool,
) -> Callable[..., Any]:
    """Create asynchronous speculative wrapper."""
    _ = max_workers  # async uses asyncio.gather, not thread pool

    @functools.wraps(fn)
    async def wrapper(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        results = await _arace_tools(tools, state)
        successful = [r for r in results if r.is_success]

        if not successful:
            _, scores = await _aselect_winner(results, evaluator)
            if fallback_on_all_fail:
                logger.warning(
                    "All %d async speculative tools failed for '%s'. Using fallback.",
                    len(results),
                    fn.__name__,
                )
                if inspect.iscoroutinefunction(fn):
                    fallback_output = await fn(state, **kwargs)
                else:
                    fallback_output = fn(state, **kwargs)
                meta = _build_race_meta(results[0], scores, results, used_fallback=True)
                return {**fallback_output, "__speculative_meta__": meta}
            meta = _build_race_meta(results[0], scores, results)
            return {"__speculative_meta__": meta}

        winner, scores = await _aselect_winner(results, evaluator)
        if not winner.is_success and successful:
            winner = successful[0]

        meta = _build_race_meta(winner, scores, results)
        output = winner.output or {}
        logger.info(
            "Async speculative node '%s': winner='%s' (score=%.3f)",
            fn.__name__,
            winner.branch_name,
            scores.get(winner.branch_name, 0.0),
        )
        return {**output, "__speculative_meta__": meta}

    return wrapper


def _race_tools_sync(
    tools: list[Callable[..., dict[str, Any]]],
    state: dict[str, Any],
    max_workers: int | None,
) -> list[BranchResult]:
    """Race all tools concurrently using a thread pool."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_tool_safe, tool, state) for tool in tools]
        return [future.result() for future in futures]


async def _arace_tools(
    tools: list[Callable[..., Any]],
    state: dict[str, Any],
) -> list[BranchResult]:
    """Race all tools concurrently using asyncio.gather."""
    tasks = [_arun_tool_safe(tool, state) for tool in tools]
    return list(await asyncio.gather(*tasks))
