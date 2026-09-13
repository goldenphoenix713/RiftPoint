"""Performance regression and benchmark assertion tests.

Validates that RiftPoint maintains high-throughput and low-latency bounds
relative to standard LangGraph checkpointers.
"""

from __future__ import annotations

import asyncio
import gc
import time
import tracemalloc
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BranchSpec,
    HeuristicEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class BenchState(TypedDict):
    """Benchmark test state schema."""

    counter: int
    data: list[str]


def create_bench_graph(saver: Any) -> Any:
    """Helper to build test graph."""
    builder = StateGraph(BenchState)

    def step_node(state: BenchState) -> dict[str, Any]:
        return {
            "counter": state["counter"] + 1,
            "data": state["data"] + ["item"],
        }

    builder.add_node("step", step_node)
    builder.add_edge(START, "step")
    return builder.compile(checkpointer=saver)


def test_put_throughput_and_latency_budget() -> None:
    """Verify RiftCheckpointSaver put operations complete within performance budgets."""
    saver = RiftCheckpointSaver()
    thread_id = "perf-test-put"
    num_writes = 200

    start_time = time.perf_counter()
    for i in range(num_writes):
        chk_id = f"1f1aef00-0000-0000-0000-{i:012d}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": chk_id,
            }
        }
        saver.put(
            config,
            {
                "v": 1,
                "id": chk_id,
                "ts": "2026-09-12T00:00:00Z",
                "channel_values": {"counter": i},
                "channel_versions": {"counter": i},
                "versions_seen": {},
                "updated_channels": ["counter"],
            },
            {"step": i},
            {"counter": i},
        )

    duration = time.perf_counter() - start_time
    avg_latency_ms = (duration / num_writes) * 1000

    # In-memory Rust-backed checkpointer should average < 1ms per put
    assert avg_latency_ms < 1.0, (
        f"Average put latency {avg_latency_ms:.3f}ms exceeded 1ms budget"
    )


def test_get_tuple_latency_budget() -> None:
    """Verify RiftCheckpointSaver get_tuple lookups complete
    within sub-millisecond budget."""
    saver = RiftCheckpointSaver()
    thread_id = "perf-test-get"

    # Seed
    for i in range(50):
        chk_id = f"1f1aef00-0000-0000-0000-{i:012d}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": chk_id,
            }
        }
        saver.put(
            config,
            {
                "v": 1,
                "id": chk_id,
                "ts": "2026-09-12T00:00:00Z",
                "channel_values": {"counter": i},
                "channel_versions": {"counter": i},
                "versions_seen": {},
                "updated_channels": ["counter"],
            },
            {"step": i},
            {"counter": i},
        )

    num_reads = 200
    start_time = time.perf_counter()
    for i in range(num_reads):
        target_id = f"1f1aef00-0000-0000-0000-{(i % 50):012d}"
        lookup_cfg: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": target_id,
            }
        }
        chk_tuple = saver.get_tuple(lookup_cfg)
        assert chk_tuple is not None

    duration = time.perf_counter() - start_time
    avg_latency_ms = (duration / num_reads) * 1000

    # Get lookups should average < 0.5ms
    assert avg_latency_ms < 0.5, (
        f"Average get latency {avg_latency_ms:.3f}ms exceeded 0.5ms budget"
    )


def test_speculative_branching_scalability_budget() -> None:
    """Verify parallel speculative branch creation scales efficiently."""
    saver = RiftCheckpointSaver()
    graph = create_bench_graph(saver)
    thread_id = "perf-test-branching"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Seed
    _ = graph.invoke({"counter": 0, "data": []}, config=config)

    runner = RiftRunner(saver=saver)
    num_branches = 20
    specs = [
        BranchSpec(name=f"perf_branch_{i}", input_data={"data": [f"spec_{i}"]})
        for i in range(num_branches)
    ]

    start_time = time.perf_counter()
    results = runner.run_parallel_branches(
        graph=graph,
        initial_config=config,
        branch_specs=specs,
    )
    duration = time.perf_counter() - start_time

    assert len(results) == num_branches
    assert all(r.is_success for r in results.values())
    # 20 concurrent branches should execute within 1 second in ThreadPool
    assert duration < 1.0, f"Branching duration {duration:.3f}s exceeded 1.0s limit"


def test_memory_saver_vs_rift_parity() -> None:
    """Verify that RiftCheckpointSaver state outputs match standard MemorySaver."""
    mem_saver = MemorySaver()
    rift_saver = RiftCheckpointSaver()

    mem_graph = create_bench_graph(mem_saver)
    rift_graph = create_bench_graph(rift_saver)

    mem_cfg: RunnableConfig = {"configurable": {"thread_id": "parity-mem"}}
    rift_cfg: RunnableConfig = {"configurable": {"thread_id": "parity-rift"}}

    mem_state = mem_graph.invoke({"counter": 0, "data": []}, config=mem_cfg)
    rift_state = rift_graph.invoke({"counter": 0, "data": []}, config=rift_cfg)

    assert mem_state["counter"] == rift_state["counter"] == 1
    assert mem_state["data"] == rift_state["data"] == ["item"]


@pytest.mark.anyio
async def test_async_put_and_get_latency_budget() -> None:
    """Verify AsyncRiftCheckpointSaver operations meet latency budgets."""
    saver = AsyncRiftCheckpointSaver()
    thread_id = "perf-async-put-get"
    num_ops = 100

    # 1. Measure aput
    t0 = time.perf_counter()
    for i in range(num_ops):
        chk_id = f"1f1aef00-0000-0000-0000-{i:012d}"
        cfg: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": chk_id,
            }
        }
        await saver.aput(
            cfg,
            {
                "v": 1,
                "id": chk_id,
                "ts": "2026-09-12T00:00:00Z",
                "channel_values": {"counter": i},
                "channel_versions": {"counter": i},
                "versions_seen": {},
                "updated_channels": ["counter"],
            },
            {"step": i},
            {"counter": i},
        )
    put_duration = time.perf_counter() - t0
    avg_put_ms = (put_duration / num_ops) * 1000
    assert avg_put_ms < 1.0, f"Async put latency {avg_put_ms:.3f}ms exceeded 1ms budget"

    # 2. Measure aget_tuple
    t1 = time.perf_counter()
    for i in range(num_ops):
        target_id = f"1f1aef00-0000-0000-0000-{i:012d}"
        lookup_cfg: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": target_id,
            }
        }
        chk_tuple = await saver.aget_tuple(lookup_cfg)
        assert chk_tuple is not None
    get_duration = time.perf_counter() - t1
    avg_get_ms = (get_duration / num_ops) * 1000
    assert avg_get_ms < 0.5, (
        f"Async get latency {avg_get_ms:.3f}ms exceeded 0.5ms budget"
    )


@pytest.mark.anyio
async def test_async_speculative_branching_budget() -> None:
    """Verify asynchronous parallel speculative branch creation and collapse."""
    saver = AsyncRiftCheckpointSaver()
    graph = create_bench_graph(saver)
    thread_id = "perf-async-branching"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Seed
    _ = await graph.ainvoke({"counter": 0, "data": []}, config=config)

    runner = RiftRunner(saver=saver)
    resolver = MultiverseResolver(saver=saver)
    num_branches = 20
    specs = [
        BranchSpec(name=f"async_branch_{i}", input_data={"data": [f"spec_{i}"]})
        for i in range(num_branches)
    ]

    start_time = time.perf_counter()
    results = await runner.arun_parallel_branches(
        graph=graph,
        initial_config=config,
        branch_specs=specs,
    )
    collapse_res = await resolver.acollapse(
        thread_id=thread_id,
        results=results,
        evaluator=HeuristicEvaluator(scorer=lambda _r: 1.0),
        prune_discarded=True,
    )
    duration = time.perf_counter() - start_time

    assert len(results) == num_branches
    assert all(r.is_success for r in results.values())
    assert collapse_res.winning_branch.startswith("async_branch_")
    assert duration < 1.0, (
        f"Async branching duration {duration:.3f}s exceeded 1.0s limit"
    )


@pytest.mark.anyio
async def test_speculative_tool_racing_simulation() -> None:
    """Verify concurrent heterogeneous tool racing speedup and collapse."""
    saver = AsyncRiftCheckpointSaver()

    class SimState(TypedDict):
        action: str
        context: list[str]
        score: float
        status: str

    builder = StateGraph(SimState)

    async def mock_tool(state: SimState) -> dict[str, Any]:
        act = state.get("action", "")
        latencies = {
            "local_cache": 0.005,
            "vector_search": 0.020,
            "sql_db": 0.025,
            "web_search": 0.040,
        }
        await asyncio.sleep(latencies.get(act, 0.01))
        scores = {
            "local_cache": 5.0,
            "vector_search": 7.5,
            "sql_db": 8.0,
            "web_search": 9.2,
        }
        return {
            "action": act,
            "context": [*state.get("context", []), f"{act}_payload"],
            "score": scores.get(act, 1.0),
            "status": "success",
        }

    builder.add_node("exec", mock_tool)
    builder.add_edge(START, "exec")
    graph = builder.compile(checkpointer=saver)

    thread_id = "test-sim-tool-race"
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(
        {"action": "seed", "context": ["init"], "score": 0.0, "status": "init"},
        config=cfg,
    )

    runner = RiftRunner(saver=saver)
    resolver = MultiverseResolver(saver=saver, branch_manager=runner.branch_manager)

    actions = ["local_cache", "vector_search", "sql_db", "web_search"]
    specs = [
        BranchSpec(name=a, input_data={"action": a, "context": [f"{a}_call"]})
        for a in actions
    ]

    t0 = time.perf_counter()
    results = await runner.arun_parallel_branches(
        graph=graph, initial_config=cfg, branch_specs=specs
    )
    collapse_res = await resolver.acollapse(
        thread_id=thread_id,
        results=results,
        evaluator=HeuristicEvaluator(
            scorer=lambda r: float(r.output.get("score", 0.0) if r.output else 0.0)
        ),
        prune_discarded=True,
    )
    dur = time.perf_counter() - t0

    assert collapse_res.winning_branch == "web_search"
    assert collapse_res.scores["web_search"].score == pytest.approx(9.2, abs=0.01)
    assert dur < 0.25
    remaining = runner.branch_manager.list_branches(thread_id)
    assert "local_cache" not in remaining
    assert "vector_search" not in remaining
    assert "sql_db" not in remaining


@pytest.mark.anyio
async def test_speculative_tool_racing_fault_tolerance() -> None:
    """Verify speculative racing gracefully ignores failed branches and auto-prunes."""
    saver = AsyncRiftCheckpointSaver()

    class FaultState(TypedDict):
        tool: str
        status: str

    builder = StateGraph(FaultState)

    def tool_exec(state: FaultState) -> dict[str, Any]:
        if state.get("tool") == "flaky_api":
            msg = "503 Service Unavailable"
            raise RuntimeError(msg)
        return {"tool": state.get("tool", ""), "status": "ok"}

    builder.add_node("exec", tool_exec)
    builder.add_edge(START, "exec")
    graph = builder.compile(checkpointer=saver)

    thread_id = "test-fault-tolerance"
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke({"tool": "seed", "status": "init"}, config=cfg)

    runner = RiftRunner(saver=saver)
    resolver = MultiverseResolver(saver=saver, branch_manager=runner.branch_manager)

    specs = [
        BranchSpec(name="flaky_branch", input_data={"tool": "flaky_api"}),
        BranchSpec(name="healthy_branch", input_data={"tool": "healthy_db"}),
    ]

    results = await runner.arun_parallel_branches(
        graph=graph, initial_config=cfg, branch_specs=specs
    )
    assert not results["flaky_branch"].is_success
    assert results["flaky_branch"].error is not None
    assert results["healthy_branch"].is_success

    collapse_res = await resolver.acollapse(
        thread_id=thread_id,
        results=results,
        evaluator=HeuristicEvaluator(scorer=lambda r: 10.0 if r.is_success else 0.0),
        prune_discarded=True,
    )

    assert collapse_res.winning_branch == "healthy_branch"
    # Ensure discarded branches were pruned from checkpoint storage
    assert "flaky_branch" not in runner.branch_manager.list_branches(thread_id)


def test_memory_and_gc_churn_budget() -> None:
    """Verify memory stability and automatic branch pruning across iterative cycles."""
    saver = RiftCheckpointSaver()
    graph = create_bench_graph(saver)
    thread_id = "test-mem-churn"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Seed initial state with ~50KB payload
    payload = "x" * 50_000
    _ = graph.invoke({"counter": 0, "data": [payload]}, config=config)

    runner = RiftRunner(saver=saver)
    resolver = MultiverseResolver(saver=saver, branch_manager=runner.branch_manager)

    tracemalloc.start()
    gc.collect()
    _init_current, init_peak = tracemalloc.get_traced_memory()

    # Run 5 iterative cycles of 10 branches each with pruning
    for turn in range(5):
        specs = [
            BranchSpec(
                name=f"t{turn}_b{b}",
                input_data={"counter": b, "data": [f"spec_{b}"]},
            )
            for b in range(10)
        ]
        results = runner.run_parallel_branches(
            graph=graph,
            initial_config=config,
            branch_specs=specs,
        )
        collapse_res = resolver.collapse(
            thread_id=thread_id,
            results=results,
            evaluator=HeuristicEvaluator(
                scorer=lambda r: float(r.output.get("counter", 0) if r.output else 0.0)
            ),
            prune_discarded=True,
        )
        assert collapse_res.winning_branch == f"t{turn}_b9"

        # Verify discarded branches were pruned
        active_branches = runner.branch_manager.list_branches(thread_id)
        assert "main" in active_branches
        assert f"t{turn}_b9" in active_branches
        for b in range(9):
            assert f"t{turn}_b{b}" not in active_branches

    gc.collect()
    _final_current, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Residual memory increase across 5 cycles should remain strictly bounded (< 5MB)
    net_increase_bytes = final_peak - init_peak
    assert net_increase_bytes < 5 * 1024 * 1024, (
        f"Memory increase {net_increase_bytes / 1024:.1f}KB exceeded 5MB threshold"
    )
