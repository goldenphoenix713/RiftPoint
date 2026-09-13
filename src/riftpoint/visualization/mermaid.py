"""Mermaid diagram generator for multiversal state DAGs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from riftpoint.checkpointer.base import BaseRiftSaver


def render_mermaid(
    saver: BaseRiftSaver,
    thread_id: str,
    *,
    direction: str = "LR",
) -> str:
    """Generate a Mermaid.js diagram representing the multiversal timeline DAG.

    Args:
        saver: The RiftPoint checkpointer backend managing multiverses.
        thread_id: Primary session thread ID to visualize.
        direction: Layout orientation ('LR' for left-to-right, 'TD' for top-down).

    Returns:
        Mermaid.js markdown syntax representing the multiverse DAG.
    """
    mv = saver.get_multiverse(thread_id)
    raw_mermaid = str(mv.visualize())

    if direction != "LR" and raw_mermaid.startswith("graph LR"):
        raw_mermaid = raw_mermaid.replace("graph LR", f"graph {direction}", 1)

    return raw_mermaid
