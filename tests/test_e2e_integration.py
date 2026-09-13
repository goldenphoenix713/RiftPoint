"""End-to-end integration tests for RiftPoint.

Validates the full lifecycle across LangGraph execution, multiversal branching,
speculative tool execution, DAG visualization, collapse, and session serde.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BranchSpec,
    ConsensusEvaluator,
    HeuristicEvaluator,
    JSONSchemaEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class ResearchState(TypedDict):
    """E2E state schema for research agent workflow."""

    query: str
    phase: str
    strategy: Literal["none", "web_search", "database", "calculator", "faulty"]
    tool_output: dict[str, Any] | str
    score: float
    history: Annotated[list[str], lambda left, right: left + right]


def query_planner_node(state: ResearchState) -> dict[str, Any]:
    """Ingest user research query and plan exploration."""
    return {
        "phase": "planned",
        "history": [f"Planned query: {state['query']}"],
    }


def web_search_tool_node(state: ResearchState) -> dict[str, Any]:
    """Execute simulated web search tool."""
    _ = state
    payload = {
        "source": "web_search",
        "answer": "Janus tachyon core provides copy-on-write timelines.",
        "confidence": 0.95,
    }
    return {
        "phase": "searched",
        "tool_output": payload,
        "score": 0.95,
        "history": ["Executed Web Search Tool"],
    }


def database_tool_node(state: ResearchState) -> dict[str, Any]:
    """Execute simulated database retrieval tool."""
    _ = state
    payload = {
        "source": "database",
        "answer": "Checkpointer storage snapshot retrieved.",
        "confidence": 0.85,
    }
    return {
        "phase": "queried",
        "tool_output": payload,
        "score": 0.85,
        "history": ["Executed Database Tool"],
    }


def calculator_tool_node(state: ResearchState) -> dict[str, Any]:
    """Execute simulated calculator tool (incomplete schema)."""
    _ = state
    payload = {
        "source": "calc",
        "value": 42,
    }
    return {
        "phase": "calculated",
        "tool_output": payload,
        "score": 0.40,
        "history": ["Executed Calculator Tool"],
    }


def faulty_tool_node(state: ResearchState) -> dict[str, Any]:
    """Simulate tool failure."""
    _ = state
    return {
        "phase": "failed",
        "tool_output": "Non-JSON failure dump",
        "score": 0.0,
        "history": ["Executed Faulty Tool"],
    }


def route_tools(
    state: ResearchState,
) -> str:
    """Conditional router for speculative tool execution."""
    strategy = state.get("strategy", "none")
    if strategy in ("web_search", "database", "calculator", "faulty"):
        return strategy
    return END


def build_research_graph() -> StateGraph[ResearchState]:
    """Construct compiled StateGraph for research agent."""
    builder = StateGraph(ResearchState)
    builder.add_node("planner", query_planner_node)
    builder.add_node("web_search", web_search_tool_node)
    builder.add_node("database", database_tool_node)
    builder.add_node("calculator", calculator_tool_node)
    builder.add_node("faulty", faulty_tool_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_tools,
        {
            "web_search": "web_search",
            "database": "database",
            "calculator": "calculator",
            "faulty": "faulty",
            END: END,
        },
    )
    builder.add_edge("web_search", END)
    builder.add_edge("database", END)
    builder.add_edge("calculator", END)
    builder.add_edge("faulty", END)

    return builder


def test_e2e_sync_lifecycle_and_session_restore() -> None:
    """Test full synchronous E2E lifecycle with multiverse collapse and restore."""
    saver = RiftCheckpointSaver()
    graph_builder = build_research_graph()
    graph = graph_builder.compile(checkpointer=saver)

    thread_id = "e2e-session-sync-01"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. Execute baseline query on canonical timeline
    initial_input: ResearchState = {
        "query": "What is the memory overhead of Janus multiversal checkpointer?",
        "phase": "init",
        "strategy": "none",
        "tool_output": "",
        "score": 0.0,
        "history": [],
    }
    canonical_state = graph.invoke(initial_input, config=config)
    assert canonical_state["phase"] == "planned"
    assert "Planned query" in canonical_state["history"][0]

    # 2. Concurrently execute 4 speculative branches
    runner = RiftRunner(saver=saver)
    specs = [
        BranchSpec(name="b_search", input_data={"strategy": "web_search"}),
        BranchSpec(name="b_db", input_data={"strategy": "database"}),
        BranchSpec(name="b_calc", input_data={"strategy": "calculator"}),
        BranchSpec(name="b_faulty", input_data={"strategy": "faulty"}),
    ]

    results = runner.run_parallel_branches(
        graph=graph,
        initial_config=config,
        branch_specs=specs,
    )

    assert len(results) == 4
    assert all(res.is_success for res in results.values())
    assert results["b_search"].output is not None
    assert results["b_search"].output["score"] == 0.95
    assert results["b_db"].output is not None
    assert results["b_db"].output["score"] == 0.85

    # 3. Mermaid Visualization
    mermaid_diag = saver.visualize(thread_id)
    assert "graph TD" in mermaid_diag
    assert thread_id not in mermaid_diag  # Thread IDs should not pollute node IDs

    # 4. Collapse Multiverse using JSONSchemaEvaluator
    schema_eval = JSONSchemaEvaluator(
        required_keys=["source", "answer", "confidence"],
        output_channel="tool_output",
    )
    resolver = MultiverseResolver(saver=saver)
    collapse_res = resolver.collapse(
        thread_id=thread_id,
        results=results,
        evaluator=schema_eval,
        prune_discarded=True,
    )

    # b_search has all 3 keys (score: 1.0)
    # b_db has all 3 keys (score: 1.0, tie-breaker picks winner alphabetically)
    # b_calc has 2 keys (source, value) -> missing answer, confidence -> score: 0.33
    # b_faulty has 0 keys -> score: 0.0
    assert collapse_res.winning_branch in ("b_search", "b_db")
    assert collapse_res.scores[collapse_res.winning_branch].score == 1.0
    assert len(collapse_res.pruned_branches) == 4

    # 5. Verify Canonical State Promotion
    canonical_tuple = saver.get_tuple(config)
    assert canonical_tuple is not None
    winner_output = collapse_res.winning_result.output
    assert winner_output is not None
    assert (
        canonical_tuple.checkpoint["channel_values"]["tool_output"]
        == winner_output["tool_output"]
    )

    # 6. Session Export & Import Restoration
    json_data = saver.export_session(thread_id)
    assert thread_id in json_data

    restored_saver = RiftCheckpointSaver()
    restored_thread_id = restored_saver.import_session(json_data)
    assert restored_thread_id == thread_id

    restored_tuple = restored_saver.get_tuple(config)
    assert restored_tuple is not None
    assert (
        restored_tuple.checkpoint["channel_values"]["tool_output"]
        == winner_output["tool_output"]
    )


def test_e2e_async_lifecycle_and_multiverse_collapse() -> None:
    """Test full asynchronous E2E lifecycle with parallel branches and collapse."""

    async def _run() -> None:
        async_saver = AsyncRiftCheckpointSaver()
        graph_builder = build_research_graph()
        graph = graph_builder.compile(checkpointer=async_saver)

        thread_id = "e2e-session-async-01"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 1. Async Canonical Baseline Execution
        initial_input: ResearchState = {
            "query": "Async multiversal exploration test.",
            "phase": "init",
            "strategy": "none",
            "tool_output": "",
            "score": 0.0,
            "history": [],
        }
        state = await graph.ainvoke(initial_input, config=config)
        assert state["phase"] == "planned"

        # 2. Async Parallel Speculative Branching
        runner = RiftRunner(saver=async_saver)
        specs = [
            BranchSpec(name="branch_a", input_data={"strategy": "web_search"}),
            BranchSpec(name="branch_b", input_data={"strategy": "calculator"}),
        ]

        results = await runner.arun_parallel_branches(
            graph=graph,
            initial_config=config,
            branch_specs=specs,
        )

        assert len(results) == 2
        assert results["branch_a"].is_success
        assert results["branch_b"].is_success

        # 3. Async Collapse with Heuristic Evaluator
        evaluator = HeuristicEvaluator(
            scorer=lambda res: (
                float(res.output.get("score", 0.0)) if res.output else 0.0
            )
        )
        resolver = MultiverseResolver(saver=async_saver)

        collapse_res = await resolver.acollapse(
            thread_id=thread_id,
            results=results,
            evaluator=evaluator,
            prune_discarded=True,
        )

        assert collapse_res.winning_branch == "branch_a"
        assert collapse_res.scores["branch_a"].score == 0.95

        # 4. Verify Final State via aget_tuple
        final_tuple = await async_saver.aget_tuple(config)
        assert final_tuple is not None
        assert final_tuple.checkpoint["channel_values"]["phase"] == "searched"

    asyncio.run(_run())


def test_e2e_multi_stage_speculative_epochs() -> None:
    """Test multi-epoch branching across successive reasoning stages."""
    saver = RiftCheckpointSaver()
    graph_builder = build_research_graph()
    graph = graph_builder.compile(checkpointer=saver)

    thread_id = "e2e-multi-stage-01"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Step 1: Initial query
    _ = graph.invoke(
        {
            "query": "Multi-stage evolution",
            "phase": "init",
            "strategy": "none",
            "tool_output": "",
            "score": 0.0,
            "history": [],
        },
        config=config,
    )

    runner = RiftRunner(saver=saver)
    resolver = MultiverseResolver(saver=saver)

    # Epoch 1: Fork candidates, evaluate, collapse
    epoch1_specs = [
        BranchSpec(name="epoch1_search", input_data={"strategy": "web_search"}),
        BranchSpec(name="epoch1_calc", input_data={"strategy": "calculator"}),
    ]
    res_epoch1 = runner.run_parallel_branches(
        graph=graph, initial_config=config, branch_specs=epoch1_specs
    )
    evaluator1 = HeuristicEvaluator(
        scorer=lambda r: float(r.output.get("score", 0.0)) if r.output else 0.0
    )
    col_epoch1 = resolver.collapse(
        thread_id=thread_id,
        results=res_epoch1,
        evaluator=evaluator1,
        prune_discarded=True,
    )
    assert col_epoch1.winning_branch == "epoch1_search"

    # Verify canonical now holds epoch 1 winning output
    tuple_e1 = saver.get_tuple(config)
    assert tuple_e1 is not None
    assert tuple_e1.checkpoint["channel_values"]["phase"] == "searched"

    # Epoch 2: Fork new candidate branches from updated canonical checkpoint
    epoch2_specs = [
        BranchSpec(name="epoch2_db", input_data={"strategy": "database"}),
        BranchSpec(name="epoch2_calc", input_data={"strategy": "calculator"}),
    ]
    res_epoch2 = runner.run_parallel_branches(
        graph=graph, initial_config=config, branch_specs=epoch2_specs
    )
    evaluator2 = HeuristicEvaluator(
        scorer=lambda r: 1.0 if r.output and r.output.get("phase") == "queried" else 0.0
    )
    col_epoch2 = resolver.collapse(
        thread_id=thread_id,
        results=res_epoch2,
        evaluator=evaluator2,
        prune_discarded=True,
    )
    assert col_epoch2.winning_branch == "epoch2_db"

    # Final verification
    tuple_e2 = saver.get_tuple(config)
    assert tuple_e2 is not None
    assert tuple_e2.checkpoint["channel_values"]["phase"] == "queried"
    assert tuple_e2.checkpoint["channel_values"]["score"] == 0.85


def test_e2e_consensus_resolution_across_branches() -> None:
    """Test consensus-based evaluation and multiverse collapse."""
    saver = RiftCheckpointSaver()
    graph_builder = build_research_graph()
    graph = graph_builder.compile(checkpointer=saver)

    thread_id = "e2e-consensus-01"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    _ = graph.invoke(
        {
            "query": "Consensus test",
            "phase": "init",
            "strategy": "none",
            "tool_output": "",
            "score": 0.0,
            "history": [],
        },
        config=config,
    )

    runner = RiftRunner(saver=saver)
    specs = [
        BranchSpec(name="cand_1", input_data={"strategy": "web_search"}),
        BranchSpec(name="cand_2", input_data={"strategy": "web_search"}),
        BranchSpec(name="cand_3", input_data={"strategy": "database"}),
    ]

    results = runner.run_parallel_branches(
        graph=graph, initial_config=config, branch_specs=specs
    )

    consensus_eval = ConsensusEvaluator(target_channel="phase")
    resolver = MultiverseResolver(saver=saver)

    collapse_res = resolver.collapse(
        thread_id=thread_id,
        results=results,
        evaluator=consensus_eval,
        prune_discarded=True,
    )

    # Both cand_1 and cand_2 have phase="searched" (majority consensus: 2/3)
    assert collapse_res.winning_branch in ("cand_1", "cand_2")
    assert collapse_res.scores[collapse_res.winning_branch].score == pytest.approx(
        2 / 3
    )


def test_e2e_time_travel_historical_checkpoint_branching() -> None:
    """Test branching from a past checkpoint ID
    (time travel), running, and collapsing."""
    saver = RiftCheckpointSaver()
    graph_builder = build_research_graph()
    graph = graph_builder.compile(checkpointer=saver)

    thread_id = "e2e-time-travel-01"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Step 1: Initial query
    _ = graph.invoke(
        {
            "query": "Step 1 Initial",
            "phase": "init",
            "strategy": "none",
            "tool_output": "",
            "score": 0.0,
            "history": [],
        },
        config=config,
    )
    earlier_tuple = saver.get_tuple(config)
    assert earlier_tuple is not None
    earlier_checkpoint_id = earlier_tuple.config["configurable"]["checkpoint_id"]

    # Step 2: Progress canonical graph further
    _ = graph.invoke(
        {
            "query": "Step 2 Advanced",
            "phase": "init",
            "strategy": "calculator",
            "tool_output": "",
            "score": 0.4,
            "history": [],
        },
        config=config,
    )
    current_tuple = saver.get_tuple(config)
    assert current_tuple is not None
    assert current_tuple.checkpoint["channel_values"]["phase"] == "calculated"

    # Time Travel: Branch from earlier historical checkpoint ID
    runner = RiftRunner(saver=saver)
    past_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": earlier_checkpoint_id,
        }
    }
    time_travel_specs = [
        BranchSpec(
            name="alternate_past_branch",
            input_data={"strategy": "web_search"},
        )
    ]
    results = runner.run_parallel_branches(
        graph=graph,
        initial_config=past_config,
        branch_specs=time_travel_specs,
    )

    assert results["alternate_past_branch"].is_success
    assert results["alternate_past_branch"].output is not None
    assert results["alternate_past_branch"].output["phase"] == "searched"

    # Collapse historical branch into canonical
    resolver = MultiverseResolver(saver=saver)
    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)
    collapse_res = resolver.collapse(
        thread_id=thread_id,
        results=results,
        evaluator=evaluator,
        prune_discarded=True,
    )
    assert collapse_res.winning_branch == "alternate_past_branch"

    # Verify canonical now reflects promoted alternate past
    final_canonical = saver.get_tuple(config)
    assert final_canonical is not None
    assert final_canonical.checkpoint["channel_values"]["phase"] == "searched"
