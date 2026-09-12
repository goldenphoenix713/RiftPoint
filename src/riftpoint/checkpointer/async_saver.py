"""Asynchronous Janus-backed CheckpointSaver for LangGraph."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    get_checkpoint_metadata,
)

from riftpoint.checkpointer.base import BaseRiftSaver

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    from langgraph.checkpoint.serde.base import SerializerProtocol


class AsyncRiftCheckpointSaver(BaseCheckpointSaver[str], BaseRiftSaver):
    """Async multiversal checkpointer for LangGraph powered by Janus-Tachyon-RS."""

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize a new AsyncRiftCheckpointSaver.

        Args:
            serde: Optional serializer for state payloads.
                   Defaults to JsonPlusSerializer.
        """
        BaseCheckpointSaver.__init__(self, serde=serde)
        BaseRiftSaver.__init__(self, serde=serde)
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, thread_id: str) -> asyncio.Lock:
        """Retrieve or create an asyncio.Lock for the specified thread ID."""
        if thread_id not in self._locks:
            self._locks[thread_id] = asyncio.Lock()
        return self._locks[thread_id]

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously retrieve a checkpoint tuple for the given configuration.

        Args:
            config: Configuration with thread_id, checkpoint_ns, and checkpoint_id.

        Returns:
            CheckpointTuple if found, None otherwise.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        raw_chk_id = get_checkpoint_metadata(config, {}).get("checkpoint_id") or config[
            "configurable"
        ].get("checkpoint_id")
        checkpoint_id = str(raw_chk_id) if raw_chk_id is not None else None

        thread_storage = self._storage.get(thread_id, {}).get(checkpoint_ns, {})
        if not thread_storage:
            return None

        target_id: str | None = checkpoint_id
        if not target_id:
            target_id = list(thread_storage.keys())[-1]

        if target_id not in thread_storage:
            return None

        storage_entry = thread_storage[target_id]
        return self._build_checkpoint_tuple(
            thread_id,
            checkpoint_ns,
            target_id,
            storage_entry,
        )

    async def _iter_ns_checkpoints_async(
        self,
        thread_id: str,
        namespace: str,
        filter_dict: dict[str, Any] | None,
        before_id: str | None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Traverse checkpoints for a namespace in reverse chronological order."""
        checkpoints = self._storage.get(thread_id, {}).get(namespace, {})
        for chk_id in reversed(list(checkpoints.keys())):
            if before_id and chk_id >= before_id:
                continue

            storage_entry = checkpoints[chk_id]
            _, ser_meta, _ = storage_entry
            meta_dict = cast(
                "CheckpointMetadata",
                self.serde.loads_typed(ser_meta),
            )
            if not self._matches_filter(meta_dict, filter_dict):
                continue

            yield self._build_checkpoint_tuple(
                thread_id,
                namespace,
                chk_id,
                storage_entry,
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoint tuples matching the specified criteria.

        Args:
            config: Optional configuration specifying thread_id and checkpoint_ns.
            filter: Optional metadata filter key-value pairs.
            before: Optional configuration specifying the upper bound checkpoint.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching CheckpointTuple instances in reverse chronological order.
        """
        thread_ids = (
            [config["configurable"]["thread_id"]]
            if config
            else list(self._storage.keys())
        )
        yielded = 0
        before_id = before["configurable"].get("checkpoint_id") if before else None

        for thread_id in thread_ids:
            ns_dict = self._storage.get(thread_id, {})
            checkpoint_ns = (
                config["configurable"].get("checkpoint_ns", "") if config else None
            )
            namespaces = (
                [checkpoint_ns] if checkpoint_ns is not None else list(ns_dict.keys())
            )

            for ns in namespaces:
                async for tuple_item in self._iter_ns_checkpoints_async(
                    thread_id, ns, filter, before_id
                ):
                    if limit is not None and yielded >= limit:
                        return
                    yield tuple_item
                    yielded += 1

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint snapshot into the Janus multiverse.

        Args:
            config: Configuration specifying thread_id and checkpoint_ns.
            checkpoint: The checkpoint payload to save.
            metadata: Associated checkpoint metadata.
            new_versions: Channel versions for updated channels.

        Returns:
            Updated RunnableConfig pointing to the new checkpoint_id.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        async with self._get_lock(thread_id):
            c_dict = dict(checkpoint)
            values = cast("dict[str, Any]", c_dict.pop("channel_values", {}))
            self._store_blobs(thread_id, checkpoint_ns, values, new_versions)

            stored_metadata = get_checkpoint_metadata(config, metadata)
            stored_chk = cast("Checkpoint", c_dict)
            self._storage[thread_id][checkpoint_ns][checkpoint_id] = (
                self.serde.dumps_typed(stored_chk),
                self.serde.dumps_typed(stored_metadata),
                parent_checkpoint_id,
            )

            # Record moment in Janus Multiverse DAG
            self._record_janus_checkpoint(
                thread_id,
                checkpoint_id,
                stored_metadata,
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate task writes for a checkpoint.

        Args:
            config: Configuration specifying thread_id, ns, and checkpoint_id.
            writes: Sequence of (channel, value) writes.
            task_id: Unique task identifier.
            task_path: Optional execution path of the task.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self._get_lock(thread_id):
            outer_key = (thread_id, checkpoint_ns, checkpoint_id)
            existing_writes = self._writes[outer_key]

            for idx, (channel, value) in enumerate(writes):
                inner_key = (task_id, WRITES_IDX_MAP.get(channel, idx))
                if inner_key[1] >= 0 and inner_key in existing_writes:
                    continue

                existing_writes[inner_key] = (
                    task_id,
                    channel,
                    self.serde.dumps_typed(value),
                    task_path,
                )

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete all checkpoints, writes, and Janus state for a thread.

        Args:
            thread_id: The thread ID to purge.
        """
        async with self._get_lock(thread_id):
            self._storage.pop(thread_id, None)
            self._multiverses.pop(thread_id, None)

            for write_key in list(self._writes.keys()):
                if write_key[0] == thread_id:
                    del self._writes[write_key]

            for blob_key in list(self._blobs.keys()):
                if blob_key[0] == thread_id:
                    del self._blobs[blob_key]

        self._locks.pop(thread_id, None)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Synchronous wrapper for get_tuple."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        raw_chk_id = get_checkpoint_metadata(config, {}).get("checkpoint_id") or config[
            "configurable"
        ].get("checkpoint_id")
        checkpoint_id = str(raw_chk_id) if raw_chk_id is not None else None

        thread_storage = self._storage.get(thread_id, {}).get(checkpoint_ns, {})
        if not thread_storage:
            return None

        target_id = checkpoint_id or list(thread_storage.keys())[-1]
        if target_id not in thread_storage:
            return None

        storage_entry = thread_storage[target_id]
        return self._build_checkpoint_tuple(
            thread_id,
            checkpoint_ns,
            target_id,
            storage_entry,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Synchronous wrapper for list."""
        thread_ids = (
            [config["configurable"]["thread_id"]]
            if config
            else list(self._storage.keys())
        )
        yielded = 0
        before_id = before["configurable"].get("checkpoint_id") if before else None

        for thread_id in thread_ids:
            ns_dict = self._storage.get(thread_id, {})
            checkpoint_ns = (
                config["configurable"].get("checkpoint_ns", "") if config else None
            )
            namespaces = (
                [checkpoint_ns] if checkpoint_ns is not None else list(ns_dict.keys())
            )

            for ns in namespaces:
                for tuple_item in self._iter_ns_checkpoints(
                    thread_id, ns, filter, before_id
                ):
                    if limit is not None and yielded >= limit:
                        return
                    yield tuple_item
                    yielded += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Synchronous wrapper for put."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        c_dict = dict(checkpoint)
        values = cast("dict[str, Any]", c_dict.pop("channel_values", {}))
        self._store_blobs(thread_id, checkpoint_ns, values, new_versions)

        stored_metadata = get_checkpoint_metadata(config, metadata)
        stored_chk = cast("Checkpoint", c_dict)
        self._storage[thread_id][checkpoint_ns][checkpoint_id] = (
            self.serde.dumps_typed(stored_chk),
            self.serde.dumps_typed(stored_metadata),
            parent_checkpoint_id,
        )

        self._record_janus_checkpoint(
            thread_id,
            checkpoint_id,
            stored_metadata,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Synchronous wrapper for put_writes."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        outer_key = (thread_id, checkpoint_ns, checkpoint_id)
        existing_writes = self._writes[outer_key]

        for idx, (channel, value) in enumerate(writes):
            inner_key = (task_id, WRITES_IDX_MAP.get(channel, idx))
            if inner_key[1] >= 0 and inner_key in existing_writes:
                continue

            existing_writes[inner_key] = (
                task_id,
                channel,
                self.serde.dumps_typed(value),
                task_path,
            )

    def delete_thread(self, thread_id: str) -> None:
        """Synchronous wrapper for delete_thread."""
        self._storage.pop(thread_id, None)
        self._multiverses.pop(thread_id, None)

        for write_key in list(self._writes.keys()):
            if write_key[0] == thread_id:
                del self._writes[write_key]

        for blob_key in list(self._blobs.keys()):
            if blob_key[0] == thread_id:
                del self._blobs[blob_key]

        self._locks.pop(thread_id, None)

    def visualize(self, thread_id: str) -> str:
        """Generate a Mermaid diagram representing the checkpoint history DAG.

        Args:
            thread_id: Thread ID to visualize.

        Returns:
            Mermaid formatted markdown string of the timeline DAG.
        """
        lines = ["```mermaid", "graph TD"]
        ns_dict = self._storage.get(thread_id, {})
        for checkpoints in ns_dict.values():
            for chk_id, (_, ser_meta, parent_id) in checkpoints.items():
                meta = cast(
                    "CheckpointMetadata",
                    self.serde.loads_typed(ser_meta),
                )
                step = meta.get("step", "")
                node_label = f'{chk_id}["{chk_id} (step: {step})"]'
                lines.append(f"    {node_label}")
                if parent_id:
                    lines.append(f"    {parent_id} --> {chk_id}")
        lines.append("```")
        return "\n".join(lines)
