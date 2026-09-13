"""Comprehensive tests for visualization (Mermaid, Plot) and session codecs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BranchManager,
    RiftCheckpointSaver,
    export_session_bytes,
    export_session_json,
    import_session_bytes,
    import_session_json,
    plot_multiverse,
    render_mermaid,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig


def test_mermaid_rendering_and_directions() -> None:
    """Test Mermaid DAG rendering with default and custom directions."""
    saver = RiftCheckpointSaver()
    manager = BranchManager(saver)
    thread_id = "test-vis-mermaid"

    config_main: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    _ = saver.put(
        config_main,
        {
            "v": 1,
            "id": "chk-1",
            "ts": "2026-09-12T00:00:00Z",
            "channel_values": {"msg": "hello"},
            "channel_versions": {"msg": 1},
            "versions_seen": {},
            "updated_channels": ["msg"],
        },
        {"step": 1},
        {"msg": 1},
    )

    _ = manager.create_branch(thread_id, "branch_exp", from_checkpoint="chk-1")

    # 1. Default LR Mermaid
    mermaid_lr = render_mermaid(saver, thread_id, direction="LR")
    assert "graph LR" in mermaid_lr

    # 2. TD orientation Mermaid
    mermaid_td = render_mermaid(saver, thread_id, direction="TD")
    assert "graph TD" in mermaid_td

    # 3. Checkpointer visualize() method
    vis_output = saver.visualize(thread_id)
    assert "```mermaid" in vis_output
    assert "chk-1" in vis_output


def test_plot_multiverse_invocation(tmp_path: Path) -> None:
    """Test graphical plot rendering and file output."""
    saver = RiftCheckpointSaver()
    thread_id = "test-plot-thread"

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    _ = saver.put(
        config,
        {
            "v": 1,
            "id": "chk-plot-1",
            "ts": "2026-09-12T00:00:00Z",
            "channel_values": {"val": 100},
            "channel_versions": {"val": 1},
            "versions_seen": {},
            "updated_channels": ["val"],
        },
        {"step": 1},
        {"val": 1},
    )

    # Invoke plot directly and through saver method
    out_file = tmp_path / "plot.png"
    plot_res1 = plot_multiverse(saver, thread_id, output_path=str(out_file))
    assert plot_res1 is not None

    plot_res2 = saver.plot(thread_id)
    assert plot_res2 is not None


def test_export_and_import_session_json() -> None:
    """Test JSON session export and restoration into a fresh saver."""
    saver = RiftCheckpointSaver()
    thread_id = "test-session-serde"

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    _ = saver.put(
        config,
        {
            "v": 1,
            "id": "chk-serde-1",
            "ts": "2026-09-12T00:00:00Z",
            "channel_values": {"data": "persisted_value"},
            "channel_versions": {"data": 1},
            "versions_seen": {},
            "updated_channels": ["data"],
        },
        {"step": 1},
        {"data": 1},
    )
    saver.put_writes(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "chk-serde-1",
            }
        },
        [("data", "write_delta")],
        task_id="task_001",
    )

    # 1. Export session
    json_str = export_session_json(saver, thread_id)
    assert "chk-serde-1" in json_str
    assert "persisted_value" not in json_str  # Base64 encoded payload

    # 2. Import into a fresh saver
    fresh_saver = RiftCheckpointSaver()
    restored_tid = import_session_json(fresh_saver, json_str)
    assert restored_tid == thread_id

    # 3. Retrieve checkpoint tuple from fresh saver
    chk_tuple = fresh_saver.get_tuple(config)
    assert chk_tuple is not None
    assert chk_tuple.checkpoint["channel_values"]["data"] == "persisted_value"
    assert chk_tuple.pending_writes is not None
    assert len(chk_tuple.pending_writes) == 1
    assert chk_tuple.pending_writes[0][1] == "data"
    assert chk_tuple.pending_writes[0][2] == "write_delta"


def test_export_and_import_session_file_and_bytes(tmp_path: Path) -> None:
    """Test file-based export/import and binary bytes codec."""
    saver = RiftCheckpointSaver()
    thread_id = "file-and-bytes-thread"

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    _ = saver.put(
        config,
        {
            "v": 1,
            "id": "chk-fb-1",
            "ts": "2026-09-12T00:00:00Z",
            "channel_values": {"counter": 42},
            "channel_versions": {"counter": 1},
            "versions_seen": {},
            "updated_channels": ["counter"],
        },
        {"step": 1},
        {"counter": 1},
    )

    # 1. File path export and import
    export_path = tmp_path / "session.json"
    saver.export_session(thread_id, file_path=str(export_path))
    assert export_path.exists()

    saver_2 = RiftCheckpointSaver()
    saver_2.import_session(str(export_path))
    tuple_2 = saver_2.get_tuple(config)
    assert tuple_2 is not None
    assert tuple_2.checkpoint["channel_values"]["counter"] == 42

    # 2. Bytes export and import
    raw_bytes = export_session_bytes(saver, thread_id)
    assert isinstance(raw_bytes, bytes)

    saver_3 = RiftCheckpointSaver()
    import_session_bytes(saver_3, raw_bytes)
    tuple_3 = saver_3.get_tuple(config)
    assert tuple_3 is not None
    assert tuple_3.checkpoint["channel_values"]["counter"] == 42


def test_async_saver_visualization_and_serde(tmp_path: Path) -> None:
    """Test async methods for visualization and session export/import."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        thread_id = "async-vis-serde-thread"

        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        await saver.aput(
            config,
            {
                "v": 1,
                "id": "chk-async-1",
                "ts": "2026-09-12T00:00:00Z",
                "channel_values": {"status": "async_active"},
                "channel_versions": {"status": 1},
                "versions_seen": {},
                "updated_channels": ["status"],
            },
            {"step": 1},
            {"status": 1},
        )

        # Plot async
        plot_obj = await saver.aplot(thread_id)
        assert plot_obj is not None

        # Export async
        file_path = tmp_path / "async_session.json"
        json_str = await saver.aexport_session(thread_id, file_path=str(file_path))
        assert "chk-async-1" in json_str

        # Import async
        fresh_saver = AsyncRiftCheckpointSaver()
        await fresh_saver.aimport_session(json_str)
        chk_tuple = await fresh_saver.aget_tuple(config)
        assert chk_tuple is not None
        assert chk_tuple.checkpoint["channel_values"]["status"] == "async_active"

    asyncio.run(_run())
