"""Comprehensive tests for BranchManager, RiftRunner, and speculative execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BranchManager,
    BranchSpec,
    RiftCheckpointSaver,
    RiftRunner,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class MultiAgentState(TypedDict):
    """Test state for multiversal agent branching."""

    messages: Annotated[list[BaseMessage], add_messages]
    strategy_used: str
    result_data: str


def test_branch_manager_lifecycle() -> None:
    """Test creating, listing, inspecting, and deleting branches."""
    saver = RiftCheckpointSaver()
    manager = BranchManager(saver)
    thread_id = "test-session-branches"

    # 1. Seed initial main checkpoint
    root_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    _ = saver.put(
        root_config,
        {
            "v": 1,
            "id": "chk-main-1",
            "ts": "2026-09-12T00:00:00Z",
            "channel_values": {"x": 1},
            "channel_versions": {"x": 1},
            "versions_seen": {},
            "updated_channels": [],
        },
        {"step": 1},
        {"x": 1},
    )

    # 2. Create branch A and branch B
    b_a_config = manager.create_branch(thread_id, "branch_sql")
    b_b_config = manager.create_branch(
        thread_id, "branch_vector", from_checkpoint="chk-main-1"
    )

    assert b_a_config["configurable"]["checkpoint_ns"] == "branch:branch_sql"
    assert b_b_config["configurable"]["checkpoint_ns"] == "branch:branch_vector"

    # 3. List branches
    branches = manager.list_branches(thread_id)
    assert "branch_sql" in branches
    assert "branch_vector" in branches

    # 4. Get branch info
    info = manager.get_branch_info(thread_id, "branch_sql")
    assert info is not None
    assert info.branch_name == "branch_sql"
    assert info.thread_id == thread_id
    assert info.checkpoint_ns == "branch:branch_sql"

    # 5. Delete branch
    manager.delete_branch(thread_id, "branch_sql")
    assert manager.get_branch_info(thread_id, "branch_sql") is None
    assert "branch_sql" not in manager.list_branches(thread_id)


def test_rift_runner_parallel_branches_sync() -> None:
    """Test synchronous parallel speculative execution across branches."""
    saver = RiftCheckpointSaver()
    runner = RiftRunner(saver)

    def tool_node(state: MultiAgentState) -> dict[str, Any]:
        strategy = state.get("strategy_used", "default")
        return {
            "messages": [AIMessage(content=f"Executed with {strategy}")],
            "result_data": f"Data for {strategy}",
        }

    builder = StateGraph(MultiAgentState)
    builder.add_node("tool_agent", tool_node)
    builder.add_edge(START, "tool_agent")
    graph = builder.compile(checkpointer=saver)

    thread_id = "speculative-sync-session"
    initial_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    # Seed main checkpoint
    graph.invoke(
        {
            "messages": [HumanMessage(content="Solve problem")],
            "strategy_used": "init",
            "result_data": "none",
        },
        config=initial_config,
    )

    # Define 3 candidate realities
    branch_specs = [
        BranchSpec(
            name="strategy_sql",
            input_data={"strategy_used": "sql_query"},
            metadata={"source": "database"},
        ),
        BranchSpec(
            name="strategy_vector",
            input_data={"strategy_used": "vector_search"},
            metadata={"source": "embeddings"},
        ),
        BranchSpec(
            name="strategy_web",
            input_data={"strategy_used": "web_scrape"},
            metadata={"source": "internet"},
        ),
    ]

    results = runner.run_parallel_branches(
        graph, initial_config, branch_specs, max_workers=3
    )

    assert len(results) == 3
    for name in ["strategy_sql", "strategy_vector", "strategy_web"]:
        res = results[name]
        assert res.is_success
        assert res.output is not None
        assert name.split("_")[1] in res.output["strategy_used"]

    # State channel isolation check:
    res_sql = results["strategy_sql"]
    assert res_sql.output is not None
    assert res_sql.output["result_data"] == "Data for sql_query"

    res_vector = results["strategy_vector"]
    assert res_vector.output is not None
    assert res_vector.output["result_data"] == "Data for vector_search"

    res_web = results["strategy_web"]
    assert res_web.output is not None
    assert res_web.output["result_data"] == "Data for web_scrape"


def test_rift_runner_parallel_branches_async() -> None:
    """Test asynchronous speculative execution with arun_parallel_branches."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        runner = RiftRunner(saver)

        async def async_node(state: MultiAgentState) -> dict[str, Any]:
            strategy = state.get("strategy_used", "default")
            await asyncio.sleep(0.001)
            return {
                "messages": [AIMessage(content=f"Async {strategy}")],
                "result_data": f"Async result {strategy}",
            }

        builder = StateGraph(MultiAgentState)
        builder.add_node("async_agent", async_node)
        builder.add_edge(START, "async_agent")
        graph = builder.compile(checkpointer=saver)

        thread_id = "speculative-async-session"
        initial_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        # Seed initial state
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Async request")],
                "strategy_used": "init",
                "result_data": "empty",
            },
            config=initial_config,
        )

        branch_specs = [
            BranchSpec(
                name="async_branch_1",
                input_data={"strategy_used": "alpha"},
            ),
            BranchSpec(
                name="async_branch_2",
                input_data={"strategy_used": "beta"},
            ),
        ]

        results = await runner.arun_parallel_branches(
            graph, initial_config, branch_specs
        )

        assert len(results) == 2
        res_1 = results["async_branch_1"]
        res_2 = results["async_branch_2"]
        assert res_1.is_success
        assert res_1.output is not None
        assert res_2.is_success
        assert res_2.output is not None
        assert res_1.output["result_data"] == "Async result alpha"
        assert res_2.output["result_data"] == "Async result beta"

    asyncio.run(_run())


def test_branch_failure_isolation() -> None:
    """Test that a failing branch does not interrupt other candidate branches."""
    saver = RiftCheckpointSaver()
    runner = RiftRunner(saver)

    def flaky_node(state: MultiAgentState) -> dict[str, Any]:
        strategy = state.get("strategy_used", "default")
        if strategy == "failing_strategy":
            msg = "Tool execution crashed!"
            raise RuntimeError(msg)
        return {
            "messages": [AIMessage(content=f"Success with {strategy}")],
            "result_data": f"OK {strategy}",
        }

    builder = StateGraph(MultiAgentState)
    builder.add_node("flaky_agent", flaky_node)
    builder.add_edge(START, "flaky_agent")
    graph = builder.compile(checkpointer=saver)

    thread_id = "error-isolation-session"
    initial_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    branch_specs = [
        BranchSpec(
            name="good_branch",
            input_data={"strategy_used": "good_strategy"},
        ),
        BranchSpec(
            name="broken_branch",
            input_data={"strategy_used": "failing_strategy"},
        ),
    ]

    results = runner.run_parallel_branches(graph, initial_config, branch_specs)

    good_res = results["good_branch"]
    assert good_res.is_success
    assert good_res.output is not None
    assert good_res.output["result_data"] == "OK good_strategy"

    broken_res = results["broken_branch"]
    assert not broken_res.is_success
    assert broken_res.error is not None
    assert isinstance(broken_res.error, RuntimeError)
    assert "Tool execution crashed!" in str(broken_res.error)
