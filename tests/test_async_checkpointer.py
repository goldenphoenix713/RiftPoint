"""Comprehensive tests for AsyncRiftCheckpointSaver."""

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

from riftpoint import AsyncRiftCheckpointSaver, BaseRiftSaver

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import RunnableConfig


class AsyncStateSchema(TypedDict):
    """Test state for async LangGraph workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


def test_async_saver_initialization() -> None:
    """Verify async saver initialization and class hierarchy."""
    saver = AsyncRiftCheckpointSaver()
    assert isinstance(saver, BaseRiftSaver)
    assert saver._storage == {}
    assert saver._multiverses == {}
    assert saver._locks == {}


def test_async_put_and_aget_tuple() -> None:
    """Test async storing and retrieving checkpoints and channel versions."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        config: RunnableConfig = {
            "configurable": {"thread_id": "async-t1", "checkpoint_ns": ""}
        }

        chk_1: Checkpoint = empty_checkpoint()
        chk_1["id"] = "chk-async-001"
        chk_1["channel_values"] = {"status": "started", "count": 10}
        chk_1["channel_versions"] = {"status": 1, "count": 1}
        meta_1: CheckpointMetadata = {"step": 1, "source": "input"}

        saved_config = await saver.aput(
            config, chk_1, meta_1, {"status": 1, "count": 1}
        )
        assert saved_config["configurable"]["checkpoint_id"] == "chk-async-001"

        retrieved = await saver.aget_tuple(config)
        assert retrieved is not None
        assert retrieved.config["configurable"]["checkpoint_id"] == "chk-async-001"
        assert retrieved.checkpoint["channel_values"]["status"] == "started"
        assert retrieved.checkpoint["channel_values"]["count"] == 10
        assert retrieved.metadata["step"] == 1

        # Second checkpoint
        chk_2: Checkpoint = empty_checkpoint()
        chk_2["id"] = "chk-async-002"
        chk_2["channel_values"] = {"status": "in_progress", "count": 20}
        chk_2["channel_versions"] = {"status": 2, "count": 2}
        meta_2: CheckpointMetadata = {"step": 2, "source": "loop"}

        await saver.aput(saved_config, chk_2, meta_2, {"status": 2, "count": 2})

        latest = await saver.aget_tuple(config)
        assert latest is not None
        assert latest.config["configurable"]["checkpoint_id"] == "chk-async-002"
        assert latest.checkpoint["channel_values"]["count"] == 20
        assert latest.parent_config is not None
        assert latest.parent_config["configurable"]["checkpoint_id"] == "chk-async-001"

        # Sync wrapper get_tuple
        sync_retrieved = saver.get_tuple(config)
        assert sync_retrieved is not None
        assert sync_retrieved.config["configurable"]["checkpoint_id"] == "chk-async-002"

    asyncio.run(_run())


def test_async_alist_and_filtering() -> None:
    """Test async listing, filtering, pagination, and limits."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        thread_id = "async-list-thread"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        for i in range(1, 6):
            chk: Checkpoint = empty_checkpoint()
            chk["id"] = f"chk-{i:03d}"
            chk["channel_values"] = {"v": i}
            chk["channel_versions"] = {"v": i}
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
            await saver.aput(parent_config, chk, meta, {"v": i})

        all_items = [c async for c in saver.alist(config)]
        assert len(all_items) == 5
        assert [c.config["configurable"]["checkpoint_id"] for c in all_items] == [
            "chk-005",
            "chk-004",
            "chk-003",
            "chk-002",
            "chk-001",
        ]

        limited = [c async for c in saver.alist(config, limit=2)]
        assert len(limited) == 2
        assert limited[0].config["configurable"]["checkpoint_id"] == "chk-005"

        filtered = [c async for c in saver.alist(config, filter={"source": "loop"})]
        assert len(filtered) == 2
        assert [c.config["configurable"]["checkpoint_id"] for c in filtered] == [
            "chk-004",
            "chk-002",
        ]

        # Sync list wrapper
        sync_items = list(saver.list(config))
        assert len(sync_items) == 5

    asyncio.run(_run())


def test_async_aput_writes_and_pending() -> None:
    """Test async pending writes buffering and restoration."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "async-writes-t",
                "checkpoint_ns": "",
                "checkpoint_id": "chk-w1",
            }
        }

        chk: Checkpoint = empty_checkpoint()
        chk["id"] = "chk-w1"
        chk["channel_values"] = {}
        chk["channel_versions"] = {}
        await saver.aput(config, chk, {"step": 1}, {})

        writes: Sequence[tuple[str, Any]] = [
            ("messages", AIMessage(content="Async AI output")),
            ("cost", 0.042),
        ]
        await saver.aput_writes(config, writes, task_id="async_worker_1")

        retrieved = await saver.aget_tuple(config)
        assert retrieved is not None
        assert retrieved.pending_writes is not None
        assert len(retrieved.pending_writes) == 2
        assert retrieved.pending_writes[0][0] == "async_worker_1"
        assert retrieved.pending_writes[0][1] == "messages"
        msg = retrieved.pending_writes[0][2]
        assert isinstance(msg, AIMessage)
        assert msg.content == "Async AI output"
        assert retrieved.pending_writes[1][2] == 0.042

        # Sync put_writes wrapper test
        sync_writes: Sequence[tuple[str, Any]] = [("cost", 0.05)]
        saver.put_writes(config, sync_writes, task_id="sync_worker")

    asyncio.run(_run())


def test_async_adelete_thread() -> None:
    """Test async purging of thread and locks."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        config: RunnableConfig = {
            "configurable": {"thread_id": "async-del-t", "checkpoint_ns": ""}
        }

        chk: Checkpoint = empty_checkpoint()
        chk["id"] = "chk-del"
        await saver.aput(config, chk, {}, {})
        saver.get_multiverse("async-del-t")

        assert await saver.aget_tuple(config) is not None
        assert "async-del-t" in saver._multiverses

        await saver.adelete_thread("async-del-t")
        assert await saver.aget_tuple(config) is None
        assert "async-del-t" not in saver._multiverses
        assert "async-del-t" not in saver._locks

        # Sync delete_thread wrapper test
        chk2: Checkpoint = empty_checkpoint()
        chk2["id"] = "chk-del-sync"
        saver.put(config, chk2, {}, {})
        assert saver.get_tuple(config) is not None
        saver.delete_thread("async-del-t")
        assert saver.get_tuple(config) is None

    asyncio.run(_run())


def test_async_concurrent_writes() -> None:
    """Test concurrent async writes with asyncio.gather across threads."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()

        async def _write_thread(thread_idx: int) -> None:
            thread_id = f"concurrent-thread-{thread_idx}"
            cfg: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
            }
            for step in range(1, 4):
                chk: Checkpoint = empty_checkpoint()
                chk["id"] = f"chk-{thread_idx}-{step}"
                chk["channel_values"] = {"step": step}
                chk["channel_versions"] = {"step": step}
                cfg = await saver.aput(cfg, chk, {"step": step}, {"step": step})
                # Small yield to encourage task switching
                await asyncio.sleep(0.001)

        await asyncio.gather(*[_write_thread(i) for i in range(5)])

        # Verify each thread recorded 3 checkpoints
        for i in range(5):
            t_cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": f"concurrent-thread-{i}",
                    "checkpoint_ns": "",
                }
            }
            chks = [c async for c in saver.alist(t_cfg)]
            assert len(chks) == 3

    asyncio.run(_run())


def test_async_visualize() -> None:
    """Test async saver Mermaid DAG visualization."""
    saver = AsyncRiftCheckpointSaver()
    thread_id = "async-viz-t"
    cfg: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    c1 = saver.put(
        cfg,
        {**empty_checkpoint(), "id": "chk-a1"},
        {"step": 1},
        {},
    )
    saver.put(
        c1,
        {**empty_checkpoint(), "id": "chk-a2"},
        {"step": 2},
        {},
    )

    mermaid_str = saver.visualize(thread_id)
    assert "```mermaid" in mermaid_str
    assert "chk-a1 --> chk-a2" in mermaid_str


def test_langgraph_ainvoke_multi_turn() -> None:
    """Test full LangGraph async execution with graph.ainvoke()."""

    async def _run() -> None:
        checkpointer = AsyncRiftCheckpointSaver()

        async def agent_node(state: AsyncStateSchema) -> dict[str, Any]:
            count = state.get("step_count", 0) + 1
            await asyncio.sleep(0.001)
            return {
                "messages": [AIMessage(content=f"Async turn {count}")],
                "step_count": count,
            }

        builder = StateGraph(AsyncStateSchema)
        builder.add_node("agent", agent_node)
        builder.add_edge(START, "agent")
        graph = builder.compile(checkpointer=checkpointer)

        thread_config: RunnableConfig = {
            "configurable": {"thread_id": "async-session-1"}
        }

        # Turn 1
        res1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hello async 1")], "step_count": 0},
            config=thread_config,
        )
        assert res1["step_count"] == 1
        assert len(res1["messages"]) == 2

        # Turn 2
        res2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hello async 2")], "step_count": 1},
            config=thread_config,
        )
        assert res2["step_count"] == 2
        assert len(res2["messages"]) == 4

        # Checkpoints in Janus
        chks = [c async for c in checkpointer.alist(thread_config)]
        assert len(chks) >= 2

    asyncio.run(_run())


def test_langgraph_astream_execution() -> None:
    """Test async streaming with graph.astream()."""

    async def _run() -> None:
        checkpointer = AsyncRiftCheckpointSaver()

        async def streaming_node(state: AsyncStateSchema) -> dict[str, Any]:
            return {
                "messages": [AIMessage(content="Stream chunk result")],
                "step_count": 1,
            }

        builder = StateGraph(AsyncStateSchema)
        builder.add_node("streamer", streaming_node)
        builder.add_edge(START, "streamer")
        graph = builder.compile(checkpointer=checkpointer)

        thread_config: RunnableConfig = {
            "configurable": {"thread_id": "stream-session"}
        }

        chunks: list[dict[str, Any]] = [
            chunk
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content="Stream test")], "step_count": 0},
                config=thread_config,
            )
        ]

        assert len(chunks) >= 1
        assert "streamer" in chunks[0]

    asyncio.run(_run())
