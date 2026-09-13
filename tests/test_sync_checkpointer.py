"""Comprehensive tests for RiftCheckpointSaver and core Janus checkpointer bridge."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import BaseRiftSaver, RiftCheckpointSaver

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import RunnableConfig


class StateSchema(TypedDict):
    """Test state for LangGraph workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


def test_saver_initialization() -> None:
    """Verify initialization and class hierarchy."""
    saver = RiftCheckpointSaver()
    assert isinstance(saver, BaseRiftSaver)
    assert saver._storage == {}
    assert saver._multiverses == {}


def test_put_and_get_tuple() -> None:
    """Test storing and retrieving single and multi-step checkpoints."""
    saver = RiftCheckpointSaver()
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    # 1. Put initial checkpoint
    chk_1: Checkpoint = empty_checkpoint()
    chk_1["id"] = "chk-001"
    chk_1["channel_values"] = {"counter": 1}
    chk_1["channel_versions"] = {"counter": 1}
    meta_1: CheckpointMetadata = {"step": 1, "source": "input"}

    saved_config = saver.put(config, chk_1, meta_1, {"counter": 1})
    assert saved_config["configurable"]["checkpoint_id"] == "chk-001"

    # 2. Retrieve checkpoint tuple
    retrieved = saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.config["configurable"]["checkpoint_id"] == "chk-001"
    assert retrieved.checkpoint["channel_values"] == {"counter": 1}
    assert retrieved.metadata["step"] == 1
    assert retrieved.parent_config is None

    # 3. Put second checkpoint with parent
    config_step2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "chk-001",
        }
    }
    chk_2: Checkpoint = empty_checkpoint()
    chk_2["id"] = "chk-002"
    chk_2["channel_values"] = {"counter": 2}
    chk_2["channel_versions"] = {"counter": 2}
    meta_2: CheckpointMetadata = {"step": 2, "source": "loop"}

    saver.put(config_step2, chk_2, meta_2, {"counter": 2})

    # Latest should now be chk-002
    latest = saver.get_tuple(config)
    assert latest is not None
    assert latest.config["configurable"]["checkpoint_id"] == "chk-002"
    assert latest.checkpoint["channel_values"] == {"counter": 2}
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == "chk-001"

    # Specific historical retrieval
    historical = saver.get_tuple(config_step2)
    assert historical is not None
    assert historical.config["configurable"]["checkpoint_id"] == "chk-001"
    assert historical.checkpoint["channel_values"] == {"counter": 1}


def test_list_checkpoints_and_filtering() -> None:
    """Test listing checkpoints, metadata filtering, and limits."""
    saver = RiftCheckpointSaver()
    thread_id = "thread-list"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    for i in range(1, 6):
        chk: Checkpoint = empty_checkpoint()
        chk["id"] = f"chk-{i:03d}"
        chk["channel_values"] = {"val": i}
        chk["channel_versions"] = {"val": i}
        meta: CheckpointMetadata = {
            "step": i,
            "source": "loop" if i % 2 == 0 else "input",
        }
        parent_config: RunnableConfig = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": f"chk-{(i - 1):03d}",
                }
            }
            if i > 1
            else config
        )
        saver.put(parent_config, chk, meta, {"val": i})

    # List all
    all_chks = list(saver.list(config))
    assert len(all_chks) == 5
    assert [c.config["configurable"]["checkpoint_id"] for c in all_chks] == [
        "chk-005",
        "chk-004",
        "chk-003",
        "chk-002",
        "chk-001",
    ]

    # Limit
    limited = list(saver.list(config, limit=2))
    assert len(limited) == 2
    assert limited[0].config["configurable"]["checkpoint_id"] == "chk-005"
    assert limited[1].config["configurable"]["checkpoint_id"] == "chk-004"

    # Filter
    evens = list(saver.list(config, filter={"source": "loop"}))
    assert len(evens) == 2
    assert [c.config["configurable"]["checkpoint_id"] for c in evens] == [
        "chk-004",
        "chk-002",
    ]

    # Global list across threads
    all_global = list(saver.list(None))
    assert len(all_global) == 5


def test_put_writes_and_pending_writes() -> None:
    """Test buffering intermediate writes and reconstructing them in CheckpointTuple."""
    saver = RiftCheckpointSaver()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-writes",
            "checkpoint_ns": "",
            "checkpoint_id": "chk-write-1",
        }
    }

    chk: Checkpoint = empty_checkpoint()
    chk["id"] = "chk-write-1"
    chk["channel_values"] = {}
    chk["channel_versions"] = {}
    saver.put(config, chk, {"step": 1}, {})

    writes: Sequence[tuple[str, Any]] = [
        ("messages", AIMessage(content="Speculative step 1")),
        ("score", 0.95),
    ]
    saver.put_writes(config, writes, task_id="node_speculative")

    retrieved = saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.pending_writes is not None
    assert len(retrieved.pending_writes) == 2
    task_ids = [w[0] for w in retrieved.pending_writes]
    channels = [w[1] for w in retrieved.pending_writes]
    assert task_ids == ["node_speculative", "node_speculative"]
    assert channels == ["messages", "score"]
    first_write = retrieved.pending_writes[0][2]
    assert isinstance(first_write, AIMessage)
    assert first_write.content == "Speculative step 1"
    assert retrieved.pending_writes[1][2] == 0.95


def test_delete_thread() -> None:
    """Test purging thread multiverse and checkpoints."""
    saver = RiftCheckpointSaver()
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-delete", "checkpoint_ns": ""}
    }

    chk: Checkpoint = empty_checkpoint()
    chk["id"] = "chk-del"
    saver.put(config, chk, {}, {})
    saver.get_multiverse("thread-delete")

    assert saver.get_tuple(config) is not None
    assert "thread-delete" in saver._multiverses

    saver.delete_thread("thread-delete")
    assert saver.get_tuple(config) is None
    assert "thread-delete" not in saver._multiverses


def test_visualize_mermaid() -> None:
    """Test generating a Mermaid DAG representation."""
    saver = RiftCheckpointSaver()
    thread_id = "thread-viz"
    root_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    c1 = saver.put(
        root_config,
        {**empty_checkpoint(), "id": "chk-1"},
        {"step": 1},
        {},
    )
    saver.put(
        c1,
        {**empty_checkpoint(), "id": "chk-2"},
        {"step": 2},
        {},
    )

    mermaid_str = saver.visualize(thread_id)
    assert "```mermaid" in mermaid_str
    assert "graph TD" in mermaid_str
    assert 'chk-1["chk-1 (step: 1)"]' in mermaid_str
    assert 'chk-2["chk-2 (step: 2)"]' in mermaid_str
    assert "chk-1 --> chk-2" in mermaid_str


def test_async_saver_methods() -> None:
    """Test asynchronous wrapper methods on RiftCheckpointSaver."""

    async def _run() -> None:
        saver = RiftCheckpointSaver()
        config: RunnableConfig = {
            "configurable": {"thread_id": "thread-async", "checkpoint_ns": ""}
        }

        chk: Checkpoint = empty_checkpoint()
        chk["id"] = "chk-async-1"
        chk["channel_values"] = {"status": "ok"}
        chk["channel_versions"] = {"status": 1}

        saved_config = await saver.aput(config, chk, {"step": 1}, {"status": 1})
        assert saved_config["configurable"]["checkpoint_id"] == "chk-async-1"

        retrieved = await saver.aget_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["status"] == "ok"

        writes: Sequence[tuple[str, Any]] = [("status", "in_progress")]
        await saver.aput_writes(saved_config, writes, task_id="async_task")

        chks = [c async for c in saver.alist(config)]
        assert len(chks) == 1

        await saver.adelete_thread("thread-async")
        assert await saver.aget_tuple(config) is None

    asyncio.run(_run())


def test_langgraph_integration_multi_turn() -> None:
    """End-to-end integration test with a compiled LangGraph StateGraph."""
    checkpointer = RiftCheckpointSaver()

    def step_node(state: StateSchema) -> dict[str, Any]:
        count = state.get("step_count", 0) + 1
        return {
            "messages": [AIMessage(content=f"Turn response {count}")],
            "step_count": count,
        }

    builder = StateGraph(StateSchema)
    builder.add_node("agent", step_node)
    builder.add_edge(START, "agent")
    graph = builder.compile(checkpointer=checkpointer)

    thread_config: RunnableConfig = {"configurable": {"thread_id": "user-session-1"}}

    # Turn 1
    input_state_1: StateSchema = {
        "messages": [HumanMessage(content="Hello 1")],
        "step_count": 0,
    }
    result_1 = graph.invoke(input_state_1, config=thread_config)
    assert result_1["step_count"] == 1
    assert len(result_1["messages"]) == 2

    # Turn 2
    input_state_2: StateSchema = {
        "messages": [HumanMessage(content="Hello 2")],
        "step_count": 1,
    }
    result_2 = graph.invoke(input_state_2, config=thread_config)
    assert result_2["step_count"] == 2
    assert len(result_2["messages"]) == 4

    # Checkpoint verification in Janus checkpointer
    checkpoints = list(checkpointer.list(thread_config))
    assert len(checkpoints) >= 2

    # Check Janus multiverse integration
    mv = checkpointer.get_multiverse("user-session-1")
    assert mv is not None
