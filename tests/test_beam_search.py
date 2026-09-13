"""Comprehensive tests for BeamSearchRunner multi-depth tree search."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BeamSearchConfig,
    BeamSearchRunner,
    BranchResult,
    BranchSpec,
    HeuristicEvaluator,
    RiftCheckpointSaver,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import RunnableConfig


class BeamState(TypedDict):
    """Test state for beam search exploration."""

    messages: Annotated[list[BaseMessage], add_messages]
    strategy: str
    score_hint: float


def _build_graph(saver: RiftCheckpointSaver | AsyncRiftCheckpointSaver) -> Any:
    """Build a simple test graph that echoes strategy and score_hint."""

    def agent_node(state: BeamState) -> dict[str, Any]:
        strategy = state.get("strategy", "default")
        score_hint = state.get("score_hint", 0.5)
        return {
            "messages": [AIMessage(content=f"Executed {strategy}")],
            "strategy": strategy,
            "score_hint": score_hint,
        }

    builder = StateGraph(BeamState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    return builder.compile(checkpointer=saver)


def _make_expand_fn(
    strategies: list[str],
) -> Any:
    """Create an expand function that produces BranchSpecs from strategy list."""

    def expand_fn(beam: BranchResult) -> Sequence[BranchSpec]:
        specs: list[BranchSpec] = []
        for i, strategy in enumerate(strategies):
            specs.append(
                BranchSpec(
                    name=f"candidate_{strategy}",
                    input_data={
                        "strategy": strategy,
                        "score_hint": float(i + 1) / len(strategies),
                    },
                )
            )
        return specs

    return expand_fn


def _score_by_hint(result: BranchResult) -> float:
    """Heuristic scorer using score_hint from state output."""
    if not result.is_success or result.output is None:
        return 0.0
    return float(result.output.get("score_hint", 0.0))


def _seed_graph(
    graph: Any,
    thread_id: str,
) -> RunnableConfig:
    """Seed initial state into the graph."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    graph.invoke(
        {
            "messages": [HumanMessage(content="Start beam search")],
            "strategy": "initial",
            "score_hint": 0.0,
        },
        config=config,
    )
    return config


def test_beam_search_basic_sync() -> None:
    """Test basic synchronous beam search with 2 depths, width 2."""
    saver = RiftCheckpointSaver()
    config = BeamSearchConfig(
        beam_width=2, branch_factor=3, max_depth=2, prune_discarded=True
    )
    evaluator = HeuristicEvaluator(scorer=_score_by_hint)
    runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

    graph = _build_graph(saver)
    thread_id = "beam-sync-basic"
    initial_config = _seed_graph(graph, thread_id)

    expand_fn = _make_expand_fn(["alpha", "beta", "gamma"])

    result = runner.run(graph, initial_config, expand_fn)

    assert result.depth_reached == 2
    assert result.total_candidates_explored > 0
    assert result.winner is not None
    assert result.winner_score > 0.0
    assert len(result.depth_history) == 2

    # Each depth should have expanded 3 candidates per beam (max 2 beams)
    for summary in result.depth_history:
        assert summary.candidates_expanded > 0
        assert len(summary.survivors) <= config.beam_width


def test_beam_search_basic_async() -> None:
    """Test basic asynchronous beam search with arun."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        config = BeamSearchConfig(
            beam_width=2, branch_factor=2, max_depth=2, prune_discarded=True
        )
        evaluator = HeuristicEvaluator(scorer=_score_by_hint)
        runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

        graph = _build_graph(saver)
        thread_id = "beam-async-basic"
        initial_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Async beam start")],
                "strategy": "init",
                "score_hint": 0.0,
            },
            config=initial_config,
        )

        expand_fn = _make_expand_fn(["x", "y"])

        result = await runner.arun(graph, initial_config, expand_fn)

        assert result.depth_reached == 2
        assert result.winner is not None
        assert result.winner_score > 0.0

    asyncio.run(_run())


def test_beam_search_pruning() -> None:
    """Verify non-surviving branches are pruned when prune_discarded=True."""
    saver = RiftCheckpointSaver()
    config = BeamSearchConfig(
        beam_width=1, branch_factor=3, max_depth=1, prune_discarded=True
    )
    evaluator = HeuristicEvaluator(scorer=_score_by_hint)
    runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

    graph = _build_graph(saver)
    thread_id = "beam-pruning"
    initial_config = _seed_graph(graph, thread_id)

    expand_fn = _make_expand_fn(["low", "mid", "high"])

    result = runner.run(graph, initial_config, expand_fn)

    assert result.depth_reached == 1
    # With beam_width=1, 2 branches should be pruned
    assert len(result.depth_history[0].pruned) == 2
    assert len(result.depth_history[0].survivors) == 1


def test_beam_search_no_pruning() -> None:
    """Verify branches persist when prune_discarded=False."""
    saver = RiftCheckpointSaver()
    config = BeamSearchConfig(
        beam_width=1, branch_factor=2, max_depth=1, prune_discarded=False
    )
    evaluator = HeuristicEvaluator(scorer=_score_by_hint)
    runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

    graph = _build_graph(saver)
    thread_id = "beam-no-prune"
    initial_config = _seed_graph(graph, thread_id)

    expand_fn = _make_expand_fn(["keep_a", "keep_b"])

    result = runner.run(graph, initial_config, expand_fn)

    assert result.depth_reached == 1
    # Pruned list is populated for metadata but branches not deleted
    assert len(result.depth_history[0].pruned) == 1
    assert len(result.depth_history[0].survivors) == 1


def test_beam_search_max_depth_termination() -> None:
    """Ensure search stops at max_depth even if candidates remain."""
    saver = RiftCheckpointSaver()
    config = BeamSearchConfig(
        beam_width=2, branch_factor=2, max_depth=3, prune_discarded=True
    )
    evaluator = HeuristicEvaluator(scorer=_score_by_hint)
    runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

    graph = _build_graph(saver)
    thread_id = "beam-max-depth"
    initial_config = _seed_graph(graph, thread_id)

    expand_fn = _make_expand_fn(["a", "b"])

    result = runner.run(graph, initial_config, expand_fn)

    assert result.depth_reached == 3
    assert len(result.depth_history) == 3


def test_beam_search_error_resilience() -> None:
    """Verify failing candidates score 0.0 and don't crash the search."""
    saver = RiftCheckpointSaver()

    def crashing_node(state: BeamState) -> dict[str, Any]:
        strategy = state.get("strategy", "default")
        if strategy == "crash":
            msg = "Intentional crash"
            raise RuntimeError(msg)
        return {
            "messages": [AIMessage(content=f"OK {strategy}")],
            "strategy": strategy,
            "score_hint": 0.8,
        }

    builder = StateGraph(BeamState)
    builder.add_node("agent", crashing_node)
    builder.add_edge(START, "agent")
    graph = builder.compile(checkpointer=saver)

    thread_id = "beam-error-resilience"
    initial_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    graph.invoke(
        {
            "messages": [HumanMessage(content="start")],
            "strategy": "init",
            "score_hint": 0.0,
        },
        config=initial_config,
    )

    def mixed_expand(beam: BranchResult) -> Sequence[BranchSpec]:
        return [
            BranchSpec(name="good", input_data={"strategy": "safe", "score_hint": 0.8}),
            BranchSpec(name="bad", input_data={"strategy": "crash", "score_hint": 0.0}),
        ]

    config = BeamSearchConfig(
        beam_width=1, branch_factor=2, max_depth=1, prune_discarded=True
    )
    evaluator = HeuristicEvaluator(scorer=_score_by_hint)
    runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

    result = runner.run(graph, initial_config, mixed_expand)

    # The search should complete without crashing
    assert result.depth_reached == 1
    assert result.winner is not None
    # The winner should be the non-crashing branch
    assert result.winner.is_success


def test_beam_search_result_metadata() -> None:
    """Verify depth_history and total_candidates_explored are accurate."""
    saver = RiftCheckpointSaver()
    config = BeamSearchConfig(
        beam_width=2, branch_factor=2, max_depth=2, prune_discarded=True
    )
    evaluator = HeuristicEvaluator(scorer=_score_by_hint)
    runner = BeamSearchRunner(saver, config=config, evaluator=evaluator)

    graph = _build_graph(saver)
    thread_id = "beam-metadata"
    initial_config = _seed_graph(graph, thread_id)

    expand_fn = _make_expand_fn(["m1", "m2"])

    result = runner.run(graph, initial_config, expand_fn)

    assert result.depth_reached == 2
    assert len(result.depth_history) == 2

    # Depth 0: 1 seed beam x 2 candidates = 2 expanded
    assert result.depth_history[0].candidates_expanded == 2
    assert result.depth_history[0].depth == 0

    # Depth 1: 2 survivors x 2 candidates = 4 expanded
    assert result.depth_history[1].candidates_expanded == 4
    assert result.depth_history[1].depth == 1

    # Total explored = 2 + 4 = 6
    assert result.total_candidates_explored == 6

    # Each depth has best_score > 0
    for summary in result.depth_history:
        assert summary.best_score > 0.0
