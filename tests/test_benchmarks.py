"""Performance regression and benchmark assertion tests.

Validates that RiftPoint maintains high-throughput and low-latency bounds
relative to standard LangGraph checkpointers.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from riftpoint import BranchSpec, RiftCheckpointSaver, RiftRunner

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
