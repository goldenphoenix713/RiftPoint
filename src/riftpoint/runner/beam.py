"""Beam search orchestration engine for multi-depth tree search over LangGraph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from riftpoint.logger import logger
from riftpoint.runner.branch import BranchManager
from riftpoint.runner.multiverse import BranchResult, BranchSpec, RiftRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

    from riftpoint.checkpointer.base import BaseRiftSaver
    from riftpoint.collapse.evaluators import BaseEvaluator

    ExpandFn = Callable[[BranchResult], Sequence[BranchSpec]]


@dataclass
class BeamSearchConfig:
    """Configuration parameters for beam search orchestration.

    Attributes:
        beam_width: K — number of parallel survivor timelines maintained
            per depth level.
        branch_factor: B — number of candidate thoughts/actions expanded
            per beam.
        max_depth: D — maximum exploration depth before terminating.
        prune_discarded: Whether to evict non-surviving branches from Janus
            DAG memory at each depth boundary.
    """

    beam_width: int = 3
    branch_factor: int = 4
    max_depth: int = 5
    prune_discarded: bool = True


@dataclass
class DepthSummary:
    """Summary of a single depth level during beam search exploration.

    Attributes:
        depth: Zero-indexed depth level.
        candidates_expanded: Number of candidate branches expanded.
        survivors: Names of surviving beams carried to the next depth.
        pruned: Names of discarded branches evicted at this depth.
        best_score: Highest evaluation score at this depth.
    """

    depth: int
    candidates_expanded: int
    survivors: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    best_score: float = 0.0


@dataclass
class BeamSearchResult:
    """Final outcome of a beam search orchestration run.

    Attributes:
        winner: The best-scoring final beam result.
        winner_score: Score of the winning beam.
        depth_reached: Actual depth explored (may be less than max_depth
            if no candidates remain).
        total_candidates_explored: Total number of branches expanded
            across all depths.
        depth_history: Per-depth summary of survivors, pruning, and scores.
    """

    winner: BranchResult
    winner_score: float
    depth_reached: int
    total_candidates_explored: int
    depth_history: list[DepthSummary] = field(default_factory=list)


class BeamSearchRunner:
    """Orchestrates multi-depth beam search over multiversal timelines.

    Iteratively expands, evaluates, selects, and prunes candidate branches
    at each depth level, composing existing RiftPoint infrastructure
    (RiftRunner, BranchManager, evaluators).
    """

    def __init__(
        self,
        saver: BaseRiftSaver,
        config: BeamSearchConfig | None = None,
        evaluator: BaseEvaluator | None = None,
        branch_manager: BranchManager | None = None,
    ) -> None:
        """Initialize the BeamSearchRunner.

        Args:
            saver: The checkpointer backend managing timeline state.
            config: Beam search configuration parameters. Defaults to
                BeamSearchConfig() with default values.
            evaluator: Evaluator instance for scoring candidate branches.
                Can also be provided per-run.
            branch_manager: Optional branch manager instance (defaults to
                new instance).
        """
        self.saver = saver
        self.config = config or BeamSearchConfig()
        self.evaluator = evaluator
        self._branch_manager = branch_manager or BranchManager(saver)
        self._runner = RiftRunner(saver, self._branch_manager)

    def run(
        self,
        graph: Pregel[Any, Any],
        initial_config: RunnableConfig,
        expand_fn: ExpandFn,
        evaluator: BaseEvaluator | None = None,
    ) -> BeamSearchResult:
        """Execute synchronous multi-depth beam search.

        Args:
            graph: Compiled LangGraph Pregel state graph.
            initial_config: Root configuration with thread_id.
            expand_fn: Callable that generates candidate BranchSpecs from a
                parent beam result.
            evaluator: Optional evaluator override for this run.

        Returns:
            BeamSearchResult with the winning beam and search metadata.

        Raises:
            ValueError: If no evaluator is provided at init or run time.
        """
        active_evaluator = evaluator or self.evaluator
        if active_evaluator is None:
            msg = "An evaluator must be provided at init or run time."
            raise ValueError(msg)

        thread_id = initial_config["configurable"]["thread_id"]
        seed_result = self._create_seed_result(thread_id)
        beams = [seed_result]
        depth_history: list[DepthSummary] = []
        total_explored = 0

        for depth in range(self.config.max_depth):
            specs = self._expand_beams(beams, expand_fn, depth)
            if not specs:
                break

            results = self._runner.run_parallel_branches(graph, initial_config, specs)
            total_explored += len(results)

            scored = self._evaluate_candidates(results, active_evaluator)
            survivors, pruned_names = self._select_survivors(scored)
            summary = self._build_depth_summary(depth, scored, survivors, pruned_names)
            depth_history.append(summary)

            if self.config.prune_discarded:
                self._prune_depth(thread_id, pruned_names)

            beams = [results[name] for name in survivors]
            if not beams:
                break

        winner, winner_score = self._pick_winner(beams, active_evaluator)

        return BeamSearchResult(
            winner=winner,
            winner_score=winner_score,
            depth_reached=len(depth_history),
            total_candidates_explored=total_explored,
            depth_history=depth_history,
        )

    async def arun(
        self,
        graph: Pregel[Any, Any],
        initial_config: RunnableConfig,
        expand_fn: ExpandFn,
        evaluator: BaseEvaluator | None = None,
    ) -> BeamSearchResult:
        """Execute asynchronous multi-depth beam search.

        Args:
            graph: Compiled LangGraph Pregel state graph.
            initial_config: Root configuration with thread_id.
            expand_fn: Callable that generates candidate BranchSpecs from a
                parent beam result.
            evaluator: Optional evaluator override for this run.

        Returns:
            BeamSearchResult with the winning beam and search metadata.

        Raises:
            ValueError: If no evaluator is provided at init or run time.
        """
        active_evaluator = evaluator or self.evaluator
        if active_evaluator is None:
            msg = "An evaluator must be provided at init or run time."
            raise ValueError(msg)

        thread_id = initial_config["configurable"]["thread_id"]
        seed_result = self._create_seed_result(thread_id)
        beams = [seed_result]
        depth_history: list[DepthSummary] = []
        total_explored = 0

        for depth in range(self.config.max_depth):
            specs = self._expand_beams(beams, expand_fn, depth)
            if not specs:
                break

            results = await self._runner.arun_parallel_branches(
                graph, initial_config, specs
            )
            total_explored += len(results)

            scored = await self._aevaluate_candidates(results, active_evaluator)
            survivors, pruned_names = self._select_survivors(scored)
            summary = self._build_depth_summary(depth, scored, survivors, pruned_names)
            depth_history.append(summary)

            if self.config.prune_discarded:
                self._prune_depth(thread_id, pruned_names)

            beams = [results[name] for name in survivors]
            if not beams:
                break

        winner, winner_score = await self._apick_winner(beams, active_evaluator)

        return BeamSearchResult(
            winner=winner,
            winner_score=winner_score,
            depth_reached=len(depth_history),
            total_candidates_explored=total_explored,
            depth_history=depth_history,
        )

    def _create_seed_result(self, thread_id: str) -> BranchResult:
        """Create a synthetic seed BranchResult representing the root state."""
        return BranchResult(
            branch_name=f"__seed_{thread_id}",
            output={},
            metadata={"seed": True},
        )

    def _expand_beams(
        self,
        beams: list[BranchResult],
        expand_fn: ExpandFn,
        depth: int,
    ) -> list[BranchSpec]:
        """Expand surviving beams into child candidate BranchSpecs.

        Args:
            beams: Current survivor beam results.
            expand_fn: User-supplied expansion function.
            depth: Current depth level (used for naming).

        Returns:
            Flattened list of child BranchSpecs.
        """
        all_specs: list[BranchSpec] = []
        for beam_idx, beam in enumerate(beams):
            try:
                children = expand_fn(beam)
                for spec in children:
                    namespaced = BranchSpec(
                        name=f"d{depth}_b{beam_idx}_{spec.name}",
                        input_data=spec.input_data,
                        metadata={
                            **spec.metadata,
                            "depth": depth,
                            "parent": beam.branch_name,
                        },
                    )
                    all_specs.append(namespaced)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Expand function failed for beam '%s' at depth %d: %s",
                    beam.branch_name,
                    depth,
                    exc,
                )
        return all_specs

    def _evaluate_candidates(
        self,
        results: dict[str, BranchResult],
        evaluator: BaseEvaluator,
    ) -> dict[str, float]:
        """Score all candidate results synchronously.

        Args:
            results: Map of branch name to BranchResult.
            evaluator: Evaluator instance.

        Returns:
            Map of branch name to score (float).
        """
        return {name: evaluator.evaluate(res).score for name, res in results.items()}

    async def _aevaluate_candidates(
        self,
        results: dict[str, BranchResult],
        evaluator: BaseEvaluator,
    ) -> dict[str, float]:
        """Score all candidate results asynchronously.

        Args:
            results: Map of branch name to BranchResult.
            evaluator: Evaluator instance.

        Returns:
            Map of branch name to score (float).
        """
        names = list(results.keys())
        tasks = [evaluator.aevaluate(results[n]) for n in names]
        eval_results = await asyncio.gather(*tasks)
        return dict(zip(names, [er.score for er in eval_results], strict=True))

    def _select_survivors(
        self,
        scored: dict[str, float],
    ) -> tuple[list[str], list[str]]:
        """Select top-K survivors and identify branches to prune.

        Args:
            scored: Map of branch name to score.

        Returns:
            Tuple of (survivor names, pruned names).
        """
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        survivors = [name for name, _ in ranked[: self.config.beam_width]]
        pruned = [name for name, _ in ranked[self.config.beam_width :]]
        return survivors, pruned

    def _prune_depth(
        self,
        thread_id: str,
        pruned_names: list[str],
    ) -> None:
        """Evict non-surviving branches from Janus DAG memory.

        Args:
            thread_id: Primary session thread ID.
            pruned_names: Branch names to delete.
        """
        for name in pruned_names:
            try:
                self._branch_manager.delete_branch(thread_id, name)
            except (KeyError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "Prune notice for branch '%s' on thread '%s': %s",
                    name,
                    thread_id,
                    exc,
                )

    def _build_depth_summary(
        self,
        depth: int,
        scored: dict[str, float],
        survivors: list[str],
        pruned: list[str],
    ) -> DepthSummary:
        """Build structured metadata for a single depth level.

        Args:
            depth: Zero-indexed depth level.
            scored: Map of branch name to score.
            survivors: Names of surviving beams.
            pruned: Names of discarded branches.

        Returns:
            DepthSummary instance.
        """
        best_score = max(scored.values()) if scored else 0.0
        logger.info(
            "Depth %d: expanded=%d, survivors=%d, pruned=%d, best=%.3f",
            depth,
            len(scored),
            len(survivors),
            len(pruned),
            best_score,
        )
        return DepthSummary(
            depth=depth,
            candidates_expanded=len(scored),
            survivors=list(survivors),
            pruned=list(pruned),
            best_score=best_score,
        )

    def _pick_winner(
        self,
        beams: list[BranchResult],
        evaluator: BaseEvaluator,
    ) -> tuple[BranchResult, float]:
        """Select the best beam from the final survivors.

        Args:
            beams: Final surviving beams.
            evaluator: Evaluator instance.

        Returns:
            Tuple of (winning BranchResult, winning score).
        """
        if not beams:
            return self._create_seed_result("__empty"), 0.0

        if len(beams) == 1:
            score = evaluator.evaluate(beams[0]).score
            return beams[0], score

        scored_beams = [(beam, evaluator.evaluate(beam).score) for beam in beams]
        scored_beams.sort(key=lambda x: x[1], reverse=True)
        return scored_beams[0]

    async def _apick_winner(
        self,
        beams: list[BranchResult],
        evaluator: BaseEvaluator,
    ) -> tuple[BranchResult, float]:
        """Asynchronously select the best beam from final survivors.

        Args:
            beams: Final surviving beams.
            evaluator: Evaluator instance.

        Returns:
            Tuple of (winning BranchResult, winning score).
        """
        if not beams:
            return self._create_seed_result("__empty"), 0.0

        if len(beams) == 1:
            eval_res = await evaluator.aevaluate(beams[0])
            return beams[0], eval_res.score

        tasks = [evaluator.aevaluate(beam) for beam in beams]
        eval_results = await asyncio.gather(*tasks)
        scored_beams = list(zip(beams, [er.score for er in eval_results], strict=True))
        scored_beams.sort(key=lambda x: x[1], reverse=True)
        return scored_beams[0]
