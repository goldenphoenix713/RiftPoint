"""Matplotlib and graphical plotting backend for RiftPoint multiverse DAGs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from riftpoint.logger import logger

if TYPE_CHECKING:
    from riftpoint.checkpointer.base import BaseRiftSaver


def plot_multiverse(
    saver: BaseRiftSaver,
    thread_id: str,
    *,
    output_path: str | None = None,
    **kwargs: Any,
) -> Any:
    """Render a graphical plot of the multiversal DAG.

    Args:
        saver: The RiftPoint checkpointer backend.
        thread_id: Primary session thread ID.
        output_path: Optional file path to save the generated figure.
        **kwargs: Additional plotting options forwarded to the Janus engine.

    Returns:
        Plot object returned by the underlying Janus engine.
    """
    mv = saver.get_multiverse(thread_id)
    plot_result = mv.plot(**kwargs)

    if output_path:
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415

            plt.savefig(output_path)
            logger.info("Saved multiverse plot to '%s'", output_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Could not save figure to '%s' via matplotlib: %s",
                output_path,
                exc,
            )

    return plot_result
