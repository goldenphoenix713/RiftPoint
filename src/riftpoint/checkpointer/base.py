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

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        source_storage = self._storage.get(thread_id, {}).get(from_namespace, {})
        if checkpoint_id in source_storage:
            entry = source_storage[checkpoint_id]
            self._storage[thread_id][to_namespace][checkpoint_id] = entry

            ser_chk, _, _ = entry
            raw_chk = self.serde.loads_typed(ser_chk)
            chk_dict = cast("Checkpoint", raw_chk)
            for channel, version in chk_dict.get("channel_versions", {}).items():
                src_key: BlobKey = (thread_id, from_namespace, channel, version)
                if src_key in self._blobs:
                    dst_key: BlobKey = (thread_id, to_namespace, channel, version)
                    self._blobs[dst_key] = self._blobs[src_key]
            return True
        return False

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
        branch_ns = f"branch:{branch_name}"
        target_id = checkpoint_id
        if not target_id:
            branch_checkpoints = self._storage.get(thread_id, {}).get(branch_ns, {})
            if branch_checkpoints:
                target_id = list(branch_checkpoints.keys())[-1]

        if not target_id or not self.copy_checkpoint_entry(
            thread_id, branch_ns, "", target_id
        ):
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
