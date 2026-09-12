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
