"""Serialization codecs and session persistence engines for RiftPoint."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from riftpoint.logger import logger

if TYPE_CHECKING:
    from riftpoint.checkpointer.base import BaseRiftSaver, BlobKey


def _bytes_to_b64(b: bytes) -> str:
    """Encode binary payload as standard base64 string."""
    return base64.b64encode(b).decode("ascii")


def _b64_to_bytes(s: str) -> bytes:
    """Decode base64 string into raw bytes."""
    return base64.b64decode(s.encode("ascii"))


@dataclass
class SessionSnapshot:
    """Portable representation of a complete multiversal session state."""

    thread_id: str
    storage: dict[str, dict[str, tuple[str, str, str, str, str | None]]]
    blobs: list[dict[str, Any]]
    writes: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def export_session_snapshot(saver: BaseRiftSaver, thread_id: str) -> SessionSnapshot:
    """Extract a complete snapshot of thread storage, blobs, and pending writes.

    Args:
        saver: Checkpointer backend containing session state.
        thread_id: Target thread ID to export.

    Returns:
        Structured SessionSnapshot dataclass.
    """
    # 1. Storage entries
    raw_storage = saver.get_session_storage(thread_id)
    formatted_storage: dict[str, dict[str, tuple[str, str, str, str, str | None]]] = {}

    for ns, chk_map in raw_storage.items():
        formatted_storage[ns] = {}
        for chk_id, (ser_chk, ser_meta, parent_id) in chk_map.items():
            formatted_storage[ns][chk_id] = (
                ser_chk[0],
                _bytes_to_b64(ser_chk[1]),
                ser_meta[0],
                _bytes_to_b64(ser_meta[1]),
                parent_id,
            )

    # 2. Blobs
    exported_blobs: list[dict[str, Any]] = []
    for (t_id, ns, channel, ver), (b_type, b_data) in saver.get_session_blobs(
        thread_id
    ):
        if t_id == thread_id:
            exported_blobs.append(
                {
                    "namespace": ns,
                    "channel": channel,
                    "version": ver,
                    "type": b_type,
                    "data": _bytes_to_b64(b_data),
                }
            )

    # 3. Writes
    exported_writes: list[dict[str, Any]] = []
    for (t_id, ns, chk_id), write_map in saver.get_session_writes(thread_id):
        if t_id == thread_id:
            for w_idx, (
                task_id,
                channel,
                (w_type, w_data),
                w_tid,
            ) in write_map.items():
                exported_writes.append(
                    {
                        "namespace": ns,
                        "checkpoint_id": chk_id,
                        "write_idx": list(w_idx),
                        "task_id": task_id,
                        "channel": channel,
                        "type": w_type,
                        "data": _bytes_to_b64(w_data),
                        "task_tid": w_tid,
                    }
                )

    return SessionSnapshot(
        thread_id=thread_id,
        storage=formatted_storage,
        blobs=exported_blobs,
        writes=exported_writes,
    )


def import_session_snapshot(saver: BaseRiftSaver, snapshot: SessionSnapshot) -> str:
    """Restore a SessionSnapshot into the checkpointer backend.

    Args:
        saver: Destination checkpointer backend.
        snapshot: SessionSnapshot instance to restore.

    Returns:
        The restored session thread ID.
    """
    thread_id = snapshot.thread_id

    # 1. Restore storage
    for ns, chk_map in snapshot.storage.items():
        for chk_id, entry in chk_map.items():
            chk_type, chk_b64, meta_type, meta_b64, parent_id = entry
            storage_entry = (
                (chk_type, _b64_to_bytes(chk_b64)),
                (meta_type, _b64_to_bytes(meta_b64)),
                parent_id,
            )
            saver.restore_storage_entry(
                thread_id=thread_id,
                namespace=ns,
                checkpoint_id=chk_id,
                entry=storage_entry,
            )

    # 2. Restore blobs
    for blob_item in snapshot.blobs:
        blob_key: BlobKey = (
            thread_id,
            blob_item["namespace"],
            blob_item["channel"],
            blob_item["version"],
        )
        blob_entry = (
            blob_item["type"],
            _b64_to_bytes(blob_item["data"]),
        )
        saver.restore_blob_entry(blob_key, blob_entry)

    # 3. Restore writes
    for write_item in snapshot.writes:
        writes_key = (
            thread_id,
            write_item["namespace"],
            write_item["checkpoint_id"],
        )
        w_idx_raw = write_item["write_idx"]
        w_idx_tuple = (str(w_idx_raw[0]), int(w_idx_raw[1]))
        write_entry = (
            write_item["task_id"],
            write_item["channel"],
            (write_item["type"], _b64_to_bytes(write_item["data"])),
            write_item["task_tid"],
        )
        saver.restore_write_entry(writes_key, w_idx_tuple, write_entry)

    logger.info("Restored session snapshot for thread '%s'", thread_id)
    return thread_id


def export_session_json(
    saver: BaseRiftSaver,
    thread_id: str,
    file_path: str | Path | None = None,
) -> str:
    """Export session state as a JSON formatted string or file.

    Args:
        saver: Checkpointer backend.
        thread_id: Session thread ID.
        file_path: Optional destination file path.

    Returns:
        JSON string representation of session snapshot.
    """
    snapshot = export_session_snapshot(saver, thread_id)
    payload = json.dumps(asdict(snapshot), indent=2)

    if file_path is not None:
        p = Path(file_path)
        p.write_text(payload, encoding="utf-8")
        logger.info("Saved session JSON to '%s'", file_path)

    return payload


def import_session_json(
    saver: BaseRiftSaver,
    data: str | Path,
) -> str:
    """Import and restore session state from JSON string or file.

    Args:
        saver: Checkpointer backend.
        data: JSON string content or Path to file.

    Returns:
        Restored thread ID.
    """
    if isinstance(data, Path):
        raw_str = data.read_text(encoding="utf-8")
    else:
        str_data = str(data).strip()
        if str_data.startswith("{"):
            raw_str = str_data
        else:
            p = Path(str_data)
            if p.exists() and p.is_file():
                raw_str = p.read_text(encoding="utf-8")
            else:
                raw_str = str_data

    data_dict = json.loads(raw_str)
    snapshot = SessionSnapshot(
        thread_id=data_dict["thread_id"],
        storage=data_dict["storage"],
        blobs=data_dict["blobs"],
        writes=data_dict["writes"],
        metadata=data_dict.get("metadata", {}),
    )
    return import_session_snapshot(saver, snapshot)


def export_session_bytes(saver: BaseRiftSaver, thread_id: str) -> bytes:
    """Export session snapshot as UTF-8 encoded JSON bytes.

    Args:
        saver: Checkpointer backend.
        thread_id: Session thread ID.

    Returns:
        Bytes payload representing the session.
    """
    json_str = export_session_json(saver, thread_id)
    return json_str.encode("utf-8")


def import_session_bytes(saver: BaseRiftSaver, payload: bytes) -> str:
    """Import session snapshot from UTF-8 encoded bytes.

    Args:
        saver: Checkpointer backend.
        payload: Binary JSON bytes.

    Returns:
        Restored thread ID.
    """
    json_str = payload.decode("utf-8")
    return import_session_json(saver, json_str)
