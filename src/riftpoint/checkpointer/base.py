"""Core bridge abstractions and shared base checkpointer for RiftPoint."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import janus
from langgraph.checkpoint.base import (
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from riftpoint.logger import logger
from riftpoint.serde.codec import export_session_json, import_session_json
from riftpoint.visualization.plot import plot_multiverse

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
    )
    from langgraph.checkpoint.serde.base import SerializerProtocol

StorageEntry = tuple[tuple[str, bytes], tuple[str, bytes], str | None]
WriteEntry = tuple[str, str, tuple[str, bytes], str]
BlobKey = tuple[str, str, str, str | int | float]


class BaseRiftSaver:
    """Shared state management bridge between LangGraph and Janus-Tachyon-RS."""

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the base RiftSaver bridge.

        Args:
            serde: Optional serializer protocol for state payloads.
                   Defaults to JsonPlusSerializer.
        """
        self.serde: SerializerProtocol = serde or JsonPlusSerializer()
        self._multiverses: dict[str, janus.MultiverseBase] = {}
        self._blobs: dict[BlobKey, tuple[str, bytes]] = {}
        self._storage: dict[str, dict[str, dict[str, StorageEntry]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._writes: dict[
            tuple[str, str, str],
            dict[tuple[str, int], WriteEntry],
        ] = defaultdict(dict)

    def get_multiverse(self, thread_id: str) -> janus.MultiverseBase:
        """Get or create the Janus MultiverseBase instance for a thread ID.

        Args:
            thread_id: The primary session thread ID.

        Returns:
            The isolated janus.MultiverseBase instance for this session.
        """
        if thread_id not in self._multiverses:
            mv = janus.MultiverseBase()
            self._multiverses[thread_id] = mv
        return self._multiverses[thread_id]

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        """Load and deserialize channel values for specified channel versions.

        Args:
            thread_id: The thread ID.
            checkpoint_ns: The checkpoint namespace.
            versions: Mapping of channel names to versions.

        Returns:
            Dictionary of deserialized channel values.
        """
        channel_values: dict[str, Any] = {}
        for channel, version in versions.items():
            blob_key: BlobKey = (thread_id, checkpoint_ns, channel, version)
            blob = self._blobs.get(blob_key)
            if blob is not None and blob[0] != "empty":
                channel_values[channel] = self.serde.loads_typed(blob)
        return channel_values

    def _store_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        new_versions: ChannelVersions,
    ) -> None:
        """Serialize and store updated channel values for new versions.

        Args:
            thread_id: The thread ID.
            checkpoint_ns: The checkpoint namespace.
            values: Channel values dictionary.
            new_versions: Updated channel versions.
        """
        for channel, version in new_versions.items():
            blob_key: BlobKey = (thread_id, checkpoint_ns, channel, version)
            if channel in values:
                self._blobs[blob_key] = self.serde.dumps_typed(values[channel])
            else:
                self._blobs[blob_key] = ("empty", b"")

    def _record_janus_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        metadata: CheckpointMetadata,
    ) -> None:
        """Record checkpoint metadata and commit into the Janus multiverse DAG.

        Args:
            thread_id: The thread ID.
            checkpoint_id: Unique checkpoint ID.
            metadata: Checkpoint metadata.
        """
        mv = self.get_multiverse(thread_id)
        step_name = f"step_{checkpoint_id}"
        try:
            tags: dict[str, Any] = {"checkpoint_id": checkpoint_id}
            if "step" in metadata:
                tags["step"] = str(metadata["step"])
            mv.tag_moment(**tags)
            mv.create_moment_label(step_name)
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.debug("Janus moment tagging skipped: %s", exc)

    def _build_checkpoint_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        storage_entry: StorageEntry,
    ) -> CheckpointTuple:
        """Construct a full CheckpointTuple from raw stored state.

        Args:
            thread_id: Thread ID.
            checkpoint_ns: Checkpoint namespace.
            checkpoint_id: Checkpoint ID.
            storage_entry: Tuple of (ser_checkpoint, ser_metadata, parent_id).

        Returns:
            Populated CheckpointTuple instance.
        """
        ser_checkpoint, ser_metadata, parent_checkpoint_id = storage_entry
        raw_chk = self.serde.loads_typed(ser_checkpoint)
        checkpoint_dict = cast("Checkpoint", raw_chk)
        raw_meta = self.serde.loads_typed(ser_metadata)
        metadata_dict = cast("CheckpointMetadata", raw_meta)
        channel_values = self._load_blobs(
            thread_id,
            checkpoint_ns,
            checkpoint_dict.get("channel_versions", {}),
        )

        writes_key = (thread_id, checkpoint_ns, checkpoint_id)
        raw_writes = self._writes.get(writes_key, {}).values()
        pending_writes = [
            (task_id, channel, self.serde.loads_typed(val))
            for task_id, channel, val, _ in raw_writes
        ]

        parent_config: RunnableConfig | None = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        full_checkpoint = cast(
            "Checkpoint",
            {
                **checkpoint_dict,
                "channel_values": channel_values,
            },
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=full_checkpoint,
            metadata=metadata_dict,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _iter_ns_checkpoints(
        self,
        thread_id: str,
        namespace: str,
        filter_dict: dict[str, Any] | None,
        before_id: str | None,
    ) -> Iterator[CheckpointTuple]:
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

    def copy_checkpoint_entry_cross_thread(
        self,
        from_thread_id: str,
        from_namespace: str,
        to_thread_id: str,
        to_namespace: str,
        checkpoint_id: str,
    ) -> bool:
        """Copy a checkpoint entry and channel blobs across threads and namespaces.

        Args:
            from_thread_id: Source session thread ID.
            from_namespace: Source namespace.
            to_thread_id: Destination session thread ID.
            to_namespace: Destination namespace.
            checkpoint_id: Checkpoint ID to copy.

        Returns:
            True if entry was found and copied, False otherwise.
        """
        source_storage = self._storage.get(from_thread_id, {}).get(from_namespace, {})
        if checkpoint_id in source_storage:
            entry = source_storage[checkpoint_id]
            if to_thread_id not in self._storage:
                self._storage[to_thread_id] = {}
            if to_namespace not in self._storage[to_thread_id]:
                self._storage[to_thread_id][to_namespace] = {}
            self._storage[to_thread_id][to_namespace][checkpoint_id] = entry

            ser_chk, _, _ = entry
            raw_chk = self.serde.loads_typed(ser_chk)
            chk_dict = cast("Checkpoint", raw_chk)
            for channel, version in chk_dict.get("channel_versions", {}).items():
                src_key: BlobKey = (
                    from_thread_id,
                    from_namespace,
                    channel,
                    version,
                )
                if src_key in self._blobs:
                    dst_key: BlobKey = (
                        to_thread_id,
                        to_namespace,
                        channel,
                        version,
                    )
                    self._blobs[dst_key] = self._blobs[src_key]
            return True
        return False

    def copy_checkpoint_entry(
        self,
        thread_id: str,
        from_namespace: str,
        to_namespace: str,
        checkpoint_id: str,
    ) -> bool:
        """Copy a checkpoint entry and channel blobs across namespaces.

        Args:
            thread_id: Session thread ID.
            from_namespace: Source namespace.
            to_namespace: Destination namespace.
            checkpoint_id: Checkpoint ID to copy.

        Returns:
            True if entry was found and copied, False otherwise.
        """
        return self.copy_checkpoint_entry_cross_thread(
            from_thread_id=thread_id,
            from_namespace=from_namespace,
            to_thread_id=thread_id,
            to_namespace=to_namespace,
            checkpoint_id=checkpoint_id,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch the checkpoint tuple matching the given configuration.

        Args:
            config: Configuration specifying thread_id, checkpoint_ns, etc.

        Returns:
            CheckpointTuple if found, None otherwise.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

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

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously fetch the checkpoint tuple matching configuration."""
        return self.get_tuple(config)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, writes, and Janus state for a thread.

        Args:
            thread_id: The thread ID to purge.
        """
        self._storage.pop(thread_id, None)
        self._multiverses.pop(thread_id, None)

        for write_key in list(self._writes.keys()):
            if write_key[0] == thread_id:
                del self._writes[write_key]

        for blob_key in list(self._blobs.keys()):
            if blob_key[0] == thread_id:
                del self._blobs[blob_key]

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete thread state."""
        self.delete_thread(thread_id)

    def delete_namespace(self, thread_id: str, namespace: str) -> None:
        """Purge a specific namespace from thread storage and channel blobs.

        Args:
            thread_id: Session thread ID.
            namespace: Namespace to delete.
        """
        if thread_id in self._storage:
            self._storage[thread_id].pop(namespace, None)

        keys_to_remove = [
            k for k in self._blobs if k[0] == thread_id and k[1] == namespace
        ]
        for k in keys_to_remove:
            self._blobs.pop(k, None)

    def commit_branch_to_canonical(
        self,
        thread_id: str,
        branch_name: str,
        checkpoint_id: str | None = None,
    ) -> RunnableConfig:
        """Promote a candidate branch state to canonical timeline (root namespace).

        Args:
            thread_id: Session thread ID.
            branch_name: The candidate branch to commit.
            checkpoint_id: Checkpoint ID to promote (defaults to latest in branch).

        Returns:
            RunnableConfig for the canonical timeline pointing to promoted checkpoint.
        """
        branch_thread_id = f"{thread_id}:branch:{branch_name}"
        branch_ns = f"branch:{branch_name}"

        target_id = checkpoint_id
        # 1. First check isolated branch_thread_id storage
        branch_storage = self._storage.get(branch_thread_id, {}).get("", {})
        if not target_id and branch_storage:
            target_id = list(branch_storage.keys())[-1]
            if not self.copy_checkpoint_entry_cross_thread(
                from_thread_id=branch_thread_id,
                from_namespace="",
                to_thread_id=thread_id,
                to_namespace="",
                checkpoint_id=target_id,
            ):
                target_id = None

        # 2. Fallback to namespaced storage within thread_id
        if not target_id:
            ns_storage = self._storage.get(thread_id, {}).get(branch_ns, {})
            if ns_storage:
                target_id = list(ns_storage.keys())[-1]
                self.copy_checkpoint_entry(
                    thread_id=thread_id,
                    from_namespace=branch_ns,
                    to_namespace="",
                    checkpoint_id=target_id,
                )

        if not target_id:
            msg = (
                f"No checkpoint found in branch '{branch_name}' "
                f"for thread '{thread_id}'"
            )
            raise ValueError(msg)

        logger.info(
            "Committed branch '%s' checkpoint '%s' to canonical timeline for '%s'",
            branch_name,
            target_id,
            thread_id,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": target_id,
            }
        }

    def get_session_storage(self, thread_id: str) -> dict[str, dict[str, StorageEntry]]:
        """Retrieve stored checkpoint entries for a thread."""
        return self._storage.get(thread_id, {})

    def get_session_blobs(
        self, thread_id: str
    ) -> list[tuple[BlobKey, tuple[str, bytes]]]:
        """Retrieve stored channel blobs for a thread."""
        return [(k, v) for k, v in self._blobs.items() if k[0] == thread_id]

    def get_session_writes(
        self, thread_id: str
    ) -> list[tuple[tuple[str, str, str], dict[tuple[str, int], WriteEntry]]]:
        """Retrieve pending task writes for a thread."""
        return [(k, v) for k, v in self._writes.items() if k[0] == thread_id]

    def restore_storage_entry(
        self,
        thread_id: str,
        namespace: str,
        checkpoint_id: str,
        entry: StorageEntry,
    ) -> None:
        """Restore a single checkpoint storage entry."""
        if thread_id not in self._storage:
            self._storage[thread_id] = {}
        if namespace not in self._storage[thread_id]:
            self._storage[thread_id][namespace] = {}
        self._storage[thread_id][namespace][checkpoint_id] = entry

    def restore_blob_entry(
        self,
        blob_key: BlobKey,
        entry: tuple[str, bytes],
    ) -> None:
        """Restore a single channel blob entry."""
        self._blobs[blob_key] = entry

    def restore_write_entry(
        self,
        writes_key: tuple[str, str, str],
        write_idx: tuple[str, int],
        entry: WriteEntry,
    ) -> None:
        """Restore a single task write entry."""
        if writes_key not in self._writes:
            self._writes[writes_key] = {}
        self._writes[writes_key][write_idx] = entry

    def plot(
        self,
        thread_id: str,
        *,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a graphical plot of the multiversal DAG.

        Args:
            thread_id: Primary session thread ID.
            output_path: Optional path to save figure.
            **kwargs: Extra plotting options forwarded to Janus engine.

        Returns:
            Plot object.
        """
        return plot_multiverse(self, thread_id, output_path=output_path, **kwargs)

    def export_session(
        self,
        thread_id: str,
        file_path: str | Path | None = None,
    ) -> str:
        """Export session state as a JSON formatted string or file.

        Args:
            thread_id: Session thread ID.
            file_path: Optional destination file path.

        Returns:
            JSON string representation of session snapshot.
        """
        return export_session_json(self, thread_id, file_path=file_path)

    def import_session(
        self,
        data: str | Path,
    ) -> str:
        """Import and restore session state from JSON string or file.

        Args:
            data: JSON string content or path to file.

        Returns:
            Restored thread ID.
        """
        return import_session_json(self, data)

    def _matches_filter(
        self,
        metadata: CheckpointMetadata,
        filter_dict: dict[str, Any] | None,
    ) -> bool:
        """Check if checkpoint metadata matches the query filter.

        Args:
            metadata: Checkpoint metadata dictionary.
            filter_dict: Query filter dictionary.

        Returns:
            True if all filter key-values match metadata, False otherwise.
        """
        if not filter_dict:
            return True
        return all(metadata.get(k) == v for k, v in filter_dict.items())
