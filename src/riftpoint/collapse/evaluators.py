"""Scoring protocols and built-in evaluators for multiverse collapse."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from riftpoint.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from riftpoint.runner.multiverse import BranchResult


@dataclass
class EvaluationResult:
    """Outcome and score from evaluating a candidate branch trajectory."""

    score: float
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_passing(self) -> bool:
        """Return True if score is positive and non-zero."""
        return self.score > 0.0


class BaseEvaluator(ABC):
    """Abstract base class for multiversal branch evaluators."""

    @abstractmethod
    def evaluate(self, result: BranchResult) -> EvaluationResult:
        """Evaluate a single candidate branch result synchronously.

        Args:
            result: The candidate branch result to score.

        Returns:
            EvaluationResult containing the numerical score and reasoning.
        """

    async def aevaluate(self, result: BranchResult) -> EvaluationResult:
        """Evaluate a single candidate branch result asynchronously.

        Args:
            result: The candidate branch result to score.

        Returns:
            EvaluationResult containing the numerical score and reasoning.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.evaluate, result)


class HeuristicEvaluator(BaseEvaluator):
    """Rule-based heuristic evaluator using custom scoring functions."""

    def __init__(
        self,
        scorer: Callable[[BranchResult], float | EvaluationResult],
        name: str = "heuristic_evaluator",
    ) -> None:
        """Initialize the HeuristicEvaluator.

        Args:
            scorer: Synchronous callable computing score or EvaluationResult.
            name: Identifier for this evaluator instance.
        """
        self.scorer = scorer
        self.name = name

    def evaluate(self, result: BranchResult) -> EvaluationResult:
        """Evaluate candidate branch result using the custom scoring callable.

        Args:
            result: Candidate branch result.

        Returns:
            Computed EvaluationResult.
        """
        if not result.is_success:
            return EvaluationResult(
                score=0.0,
                reasoning=f"Branch execution failed with error: {result.error}",
                metadata={"evaluator": self.name, "error": str(result.error)},
            )

        try:
            score_or_res = self.scorer(result)
            if isinstance(score_or_res, EvaluationResult):
                return score_or_res
            return EvaluationResult(
                score=float(score_or_res),
                metadata={"evaluator": self.name},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Heuristic evaluation error for branch '%s': %s",
                result.branch_name,
                exc,
            )
            return EvaluationResult(
                score=0.0,
                reasoning=f"Evaluation failed: {exc}",
                metadata={"evaluator": self.name, "error": str(exc)},
            )


class JSONSchemaEvaluator(BaseEvaluator):
    """Evaluates whether candidate branch output conforms to required keys/schema."""

    def __init__(
        self,
        required_keys: Sequence[str],
        output_channel: str | None = None,
        name: str = "json_schema_evaluator",
    ) -> None:
        """Initialize the JSONSchemaEvaluator.

        Args:
            required_keys: Collection of keys that must be present in output.
            output_channel: Optional specific state channel containing dict
                output.
            name: Identifier for this evaluator instance.
        """
        self.required_keys = list(required_keys)
        self.output_channel = output_channel
        self.name = name

    def _extract_target_dict(
        self, result: BranchResult
    ) -> tuple[dict[str, Any] | None, EvaluationResult | None]:
        """Extract target dictionary from branch output."""
        if not result.is_success or result.output is None:
            return None, EvaluationResult(
                score=0.0,
                reasoning="Branch did not produce valid output",
                metadata={"evaluator": self.name},
            )

        target_dict = result.output
        if self.output_channel is not None:
            val = result.output.get(self.output_channel)
            if not isinstance(val, dict):
                return None, EvaluationResult(
                    score=0.0,
                    reasoning=f"Channel '{self.output_channel}' is not a dictionary",
                    metadata={"evaluator": self.name},
                )
            target_dict = val
        return target_dict, None

    def evaluate(self, result: BranchResult) -> EvaluationResult:
        """Score branch based on presence of required fields.

        Args:
            result: Candidate branch result.

        Returns:
            EvaluationResult with ratio of satisfied keys (0.0 to 1.0).
        """
        target_dict, err_res = self._extract_target_dict(result)
        if err_res is not None or target_dict is None:
            return err_res or EvaluationResult(score=0.0)

        if not self.required_keys:
            return EvaluationResult(
                score=1.0,
                reasoning="No schema constraints defined",
                metadata={"evaluator": self.name},
            )

        present_keys = [k for k in self.required_keys if k in target_dict]
        missing_keys = [k for k in self.required_keys if k not in target_dict]
        score = len(present_keys) / len(self.required_keys)

        reasoning = (
            "All required keys present"
            if score == 1.0
            else f"Missing required keys: {missing_keys}"
        )
        return EvaluationResult(
            score=score,
            reasoning=reasoning,
            metadata={
                "evaluator": self.name,
                "present_keys": present_keys,
                "missing_keys": missing_keys,
            },
        )


class ConsensusEvaluator(BaseEvaluator):
    """Evaluates branches against consensus modal agreement across candidate pool."""

    def __init__(
        self,
        target_channel: str,
        name: str = "consensus_evaluator",
    ) -> None:
        """Initialize the ConsensusEvaluator.

        Args:
            target_channel: Channel key whose value is compared across branches.
            name: Identifier for this evaluator instance.
        """
        self.target_channel = target_channel
        self.name = name
        self._consensus_value: Any = None
        self._consensus_ratio: float = 1.0

    def fit(self, results: Sequence[BranchResult]) -> None:
        """Analyze pool of branch results to determine the consensus modal value.

        Args:
            results: Sequence of candidate branch results.
        """
        valid_values = [
            res.output[self.target_channel]
            for res in results
            if res.is_success and res.output and self.target_channel in res.output
        ]

        if not valid_values:
            self._consensus_value = None
            self._consensus_ratio = 0.0
            return

        counts = Counter(valid_values)
        most_common_val, most_common_count = counts.most_common(1)[0]
        self._consensus_value = most_common_val
        self._consensus_ratio = most_common_count / len(valid_values)

    def evaluate(self, result: BranchResult) -> EvaluationResult:
        """Score branch based on match with consensus value.

        Args:
            result: Candidate branch result.

        Returns:
            EvaluationResult with consensus score.
        """
        if not result.is_success or result.output is None:
            return EvaluationResult(
                score=0.0,
                reasoning="Branch did not complete successfully",
                metadata={"evaluator": self.name},
            )

        val = result.output.get(self.target_channel)
        if val == self._consensus_value:
            return EvaluationResult(
                score=self._consensus_ratio,
                reasoning=f"Matches consensus value for '{self.target_channel}'",
                metadata={
                    "evaluator": self.name,
                    "target_channel": self.target_channel,
                    "consensus_value": str(self._consensus_value),
                },
            )

        return EvaluationResult(
            score=0.0,
            reasoning=f"Diverges from consensus value for '{self.target_channel}'",
            metadata={
                "evaluator": self.name,
                "branch_value": str(val),
                "consensus_value": str(self._consensus_value),
            },
        )


class LLMJudgeEvaluator(BaseEvaluator):
    """LLM-as-a-judge scoring evaluator using customizable prompt rubrics."""

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        rubric_prompt: str | None = None,
        name: str = "llm_judge_evaluator",
    ) -> None:
        """Initialize the LLMJudgeEvaluator.

        Args:
            judge_fn: Callable taking formatted evaluation prompt and returning
                LLM response text.
            rubric_prompt: Optional custom evaluation rubric template.
            name: Identifier for this evaluator instance.
        """
        self.judge_fn = judge_fn
        self.rubric_prompt = rubric_prompt or (
            "Evaluate agent output on a scale of 0.0 to 1.0.\n"
            "Format response as 'SCORE: <float>\\nREASONING: <text>'\n\n"
            "Candidate Output:\n{output}"
        )
        self.name = name

    def evaluate(self, result: BranchResult) -> EvaluationResult:
        """Score candidate branch output using the LLM judge function.

        Args:
            result: Candidate branch result.

        Returns:
            EvaluationResult parsed from LLM judge response.
        """
        if not result.is_success or result.output is None:
            return EvaluationResult(
                score=0.0,
                reasoning="Branch execution failed",
                metadata={"evaluator": self.name},
            )

        prompt = self.rubric_prompt.format(output=str(result.output))
        try:
            raw_response = self.judge_fn(prompt)
            score, reasoning = self._parse_llm_response(raw_response)
            return EvaluationResult(
                score=score,
                reasoning=reasoning,
                metadata={"evaluator": self.name, "raw_response": raw_response},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "LLM judge evaluation failed for '%s': %s",
                result.branch_name,
                exc,
            )
            return EvaluationResult(
                score=0.0,
                reasoning=f"Judge error: {exc}",
                metadata={"evaluator": self.name, "error": str(exc)},
            )

    def _parse_llm_response(self, text: str) -> tuple[float, str]:
        """Extract score float and reasoning from LLM text output."""
        lines = text.strip().splitlines()
        score = 0.5
        reasoning_lines: list[str] = []

        for line in lines:
            line_clean = line.strip()
            if line_clean.upper().startswith("SCORE:"):
                score_str = line_clean.split(":", 1)[1].strip()
                try:
                    score = float(score_str)
                except ValueError as exc:
                    logger.debug("Failed parsing score '%s': %s", score_str, exc)
            elif line_clean.upper().startswith("REASONING:"):
                reasoning_lines.append(line_clean.split(":", 1)[1].strip())
            else:
                reasoning_lines.append(line_clean)

        reasoning = (
            " ".join(reasoning_lines) if reasoning_lines else "Evaluated by LLM Judge"
        )
        return min(max(score, 0.0), 1.0), reasoning
