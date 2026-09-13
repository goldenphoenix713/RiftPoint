"""Multiverse collapse and canonical timeline resolution engine for RiftPoint."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from riftpoint.collapse.evaluators import ConsensusEvaluator
from riftpoint.logger import logger
from riftpoint.runner.branch import BranchManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig

    from riftpoint.checkpointer.base import BaseRiftSaver
    from riftpoint.collapse.evaluators import BaseEvaluator, EvaluationResult
    from riftpoint.runner.multiverse import BranchResult


@dataclass
class CollapseResult:
    """Outcome of collapsing candidate branches into the canonical timeline."""

    winning_branch: str
    winning_result: BranchResult
    scores: dict[str, EvaluationResult]
    canonical_config: RunnableConfig
    pruned_branches: list[str] = field(default_factory=list)


class MultiverseResolver:
    """Evaluates candidate realities, commits winner, and prunes timelines."""

    def __init__(
        self,
        saver: BaseRiftSaver,
        branch_manager: BranchManager | None = None,
    ) -> None:
        """Initialize the MultiverseResolver.

        Args:
            saver: RiftPoint checkpointer backend.
            branch_manager: Optional branch manager instance.
        """
        self.saver = saver
        self.branch_manager = branch_manager or BranchManager(saver)

    def evaluate_branches(
        self,
        results: dict[str, BranchResult],
        evaluator: BaseEvaluator,
    ) -> dict[str, EvaluationResult]:
        """Evaluate candidate branch results synchronously.

        Args:
            results: Map of branch name to BranchResult.
            evaluator: Evaluator instance.

        Returns:
            Map of branch name to EvaluationResult.
        """
        if isinstance(evaluator, ConsensusEvaluator):
            evaluator.fit(list(results.values()))

        scores: dict[str, EvaluationResult] = {}
        for branch_name, res in results.items():
            scores[branch_name] = evaluator.evaluate(res)
        return scores

    async def aevaluate_branches(
        self,
        results: dict[str, BranchResult],
        evaluator: BaseEvaluator,
    ) -> dict[str, EvaluationResult]:
        """Evaluate candidate branch results asynchronously.

        Args:
            results: Map of branch name to BranchResult.
            evaluator: Evaluator instance.

        Returns:
            Map of branch name to EvaluationResult.
        """
        if isinstance(evaluator, ConsensusEvaluator):
            evaluator.fit(list(results.values()))

        branch_names = list(results.keys())
        tasks = [evaluator.aevaluate(results[name]) for name in branch_names]
        evaluation_results = await asyncio.gather(*tasks)
        return dict(zip(branch_names, evaluation_results, strict=True))

    def _get_target_scores(
        self,
        scores: dict[str, EvaluationResult],
        results: dict[str, BranchResult],
    ) -> dict[str, float]:
        """Filter target pool prioritizing successful branches."""
        success_scores = {
            name: ev.score
            for name, ev in scores.items()
            if results.get(name) and results[name].is_success
        }
        return success_scores or {name: ev.score for name, ev in scores.items()}

    def select_winner(
        self,
        scores: dict[str, EvaluationResult],
        results: dict[str, BranchResult],
        tie_breaker: Callable[[list[str]], str] | None = None,
    ) -> str:
        """Determine the winning candidate branch based on evaluation scores.

        Args:
            scores: Map of branch name to EvaluationResult.
            results: Map of branch name to BranchResult.
            tie_breaker: Optional callback resolving ties among highest scores.

        Returns:
            The winning branch name.

        Raises:
            ValueError: If no candidate branches are provided.
        """
        if not scores:
            msg = "Cannot select winner from empty branch results."
            raise ValueError(msg)

        target_pool = self._get_target_scores(scores, results)
        max_score = max(target_pool.values())
        top_candidates = [
            name for name, score in target_pool.items() if score == max_score
        ]

        if len(top_candidates) == 1:
            return top_candidates[0]

        if tie_breaker:
            return tie_breaker(top_candidates)

        return min(top_candidates)

    def collapse(
        self,
        thread_id: str,
        results: dict[str, BranchResult],
        evaluator: BaseEvaluator,
        *,
        prune_discarded: bool = True,
        tie_breaker: Callable[[list[str]], str] | None = None,
    ) -> CollapseResult:
        """Collapse candidate branches and commit winning state to canonical.

        Args:
            thread_id: Primary session thread ID.
            results: Map of candidate branch results.
            evaluator: Evaluator instance scoring branches.
            prune_discarded: Whether to delete discarded candidate branches.
            tie_breaker: Optional callback resolving ties.

        Returns:
            CollapseResult describing winner, scores, config, and pruned branches.
        """
        scores = self.evaluate_branches(results, evaluator)
        winner_name = self.select_winner(scores, results, tie_breaker)
        winning_result = results[winner_name]

        # 1. Commit winning branch to canonical timeline
        canonical_config = self.saver.commit_branch_to_canonical(
            thread_id=thread_id,
            branch_name=winner_name,
        )

        # 2. Prune discarded candidate branches if requested
        pruned_branches: list[str] = []
        if prune_discarded:
            for branch_name in list(results.keys()):
                self.branch_manager.delete_branch(thread_id, branch_name)
                pruned_branches.append(branch_name)

        logger.info(
            "Collapsed multiverse for thread '%s'. Winner: '%s' (score: %.3f)",
            thread_id,
            winner_name,
            scores[winner_name].score,
        )

        return CollapseResult(
            winning_branch=winner_name,
            winning_result=winning_result,
            scores=scores,
            canonical_config=canonical_config,
            pruned_branches=pruned_branches,
        )

    async def acollapse(
        self,
        thread_id: str,
        results: dict[str, BranchResult],
        evaluator: BaseEvaluator,
        *,
        prune_discarded: bool = True,
        tie_breaker: Callable[[list[str]], str] | None = None,
    ) -> CollapseResult:
        """Asynchronously collapse candidate branches and commit winning state.

        Args:
            thread_id: Primary session thread ID.
            results: Map of candidate branch results.
            evaluator: Evaluator instance scoring branches.
            prune_discarded: Whether to delete discarded candidate branches.
            tie_breaker: Optional callback resolving ties.

        Returns:
            CollapseResult describing winner, scores, config, and pruned branches.
        """
        scores = await self.aevaluate_branches(results, evaluator)
        winner_name = self.select_winner(scores, results, tie_breaker)
        winning_result = results[winner_name]

        # 1. Commit winning branch to canonical timeline
        canonical_config = self.saver.commit_branch_to_canonical(
            thread_id=thread_id,
            branch_name=winner_name,
        )

        # 2. Prune discarded candidate branches if requested
        pruned_branches: list[str] = []
        if prune_discarded:
            for branch_name in list(results.keys()):
                self.branch_manager.delete_branch(thread_id, branch_name)
                pruned_branches.append(branch_name)

        logger.info(
            "Async collapsed multiverse for thread '%s'. Winner: '%s' (score: %.3f)",
            thread_id,
            winner_name,
            scores[winner_name].score,
        )

        return CollapseResult(
            winning_branch=winner_name,
            winning_result=winning_result,
            scores=scores,
            canonical_config=canonical_config,
            pruned_branches=pruned_branches,
        )
