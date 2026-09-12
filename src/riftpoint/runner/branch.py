"""Branch management and timeline lifecycle engine for RiftPoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from riftpoint.logger import logger

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from riftpoint.checkpointer.base import BaseRiftSaver


@dataclass
class BranchInfo:
    """Metadata describing a candidate branch in a session multiverse."""

    branch_name: str
    thread_id: str
    parent_checkpoint_id: str | None
    checkpoint_ns: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class BranchManager:
    """Orchestrates candidate branch creation, switching, and pruning."""

    def __init__(self, saver: BaseRiftSaver) -> None:
        """Initialize the BranchManager with a checkpointer backend.

        Args:
            saver: The RiftPoint checkpointer backend managing the session multiverses.
        """
        self.saver = saver
        self._branches: dict[str, dict[str, BranchInfo]] = {}

    def create_branch(
        self,
        thread_id: str,
        branch_name: str,
        from_checkpoint: str | None = None,
    ) -> RunnableConfig:
        """Fork execution from a designated checkpoint into a candidate timeline.

        Args:
            thread_id: The primary session thread ID.
            branch_name: Unique name for the candidate branch.
            from_checkpoint: Optional checkpoint ID to fork from
                (defaults to latest head).

        Returns:
            RunnableConfig configured for the new candidate branch.
        """
        checkpoint_ns = f"branch:{branch_name}"
        mv = self.saver.get_multiverse(thread_id)

        # 1. Resolve parent checkpoint
        parent_id = from_checkpoint
        if not parent_id:
            root_tuple = self.saver.get_tuple(  # type: ignore[attr-defined]
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            )
            if root_tuple is not None:
                parent_id = root_tuple.config["configurable"].get("checkpoint_id")

        # 2. Register branch in Janus Multiverse
        try:
            if branch_name not in mv.list_branches():
                mv.create_branch(branch_name)
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Janus create_branch notice for %s on %s: %s",
                branch_name,
                thread_id,
                exc,
            )

        # 3. Seed the branch namespace in checkpointer storage if parent exists
        if parent_id:
            self.saver.copy_checkpoint_entry(
                thread_id=thread_id,
                from_namespace="",
                to_namespace=checkpoint_ns,
                checkpoint_id=parent_id,
            )

        # 4. Record branch metadata
        if thread_id not in self._branches:
            self._branches[thread_id] = {}

        info = BranchInfo(
            branch_name=branch_name,
            thread_id=thread_id,
            parent_checkpoint_id=parent_id,
            checkpoint_ns=checkpoint_ns,
        )
        self._branches[thread_id][branch_name] = info

        logger.info(
            "Created branch '%s' for thread '%s' from checkpoint '%s'",
            branch_name,
            thread_id,
            parent_id,
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_id,
            }
        }
        return config

    def list_branches(self, thread_id: str) -> list[str]:
        """List all active branch names for a session thread.

        Args:
            thread_id: Primary session thread ID.

        Returns:
            List of branch names.
        """
        branches = set(self._branches.get(thread_id, {}).keys())
        mv = self.saver.get_multiverse(thread_id)
        try:
            branches.update(mv.list_branches())
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Janus list_branches notice for %s: %s",
                thread_id,
                exc,
            )
        return sorted(branches)

    def get_branch_info(self, thread_id: str, branch_name: str) -> BranchInfo | None:
        """Retrieve metadata for a candidate branch.

        Args:
            thread_id: Session thread ID.
            branch_name: Branch name.

        Returns:
            BranchInfo if found, None otherwise.
        """
        return self._branches.get(thread_id, {}).get(branch_name)

    def delete_branch(self, thread_id: str, branch_name: str) -> None:
        """Purge a branch timeline and its storage from the session multiverse.

        Args:
            thread_id: Session thread ID.
            branch_name: Branch name to delete.
        """
        checkpoint_ns = f"branch:{branch_name}"
        self.saver.delete_namespace(thread_id, checkpoint_ns)

        if thread_id in self._branches:
            self._branches[thread_id].pop(branch_name, None)

        mv = self.saver.get_multiverse(thread_id)
        try:
            if branch_name in mv.list_branches():
                mv.delete_branch(branch_name)
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Janus delete_branch notice for %s on %s: %s",
                branch_name,
                thread_id,
                exc,
            )

        logger.info("Deleted branch '%s' for thread '%s'", branch_name, thread_id)
