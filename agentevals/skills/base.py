"""Abstract base class for all agentevals evaluation skills."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentevals.models import EvalInput, SkillResult


class BaseSkill(ABC):
    """Base class that all evaluation skills must subclass.

    A *skill* encapsulates a single dimension of evaluation (e.g., correctness,
    relevance).  Skills are registered with an :class:`~agentevals.agent.EvalsAgent`
    and invoked during :py:meth:`~agentevals.agent.EvalsAgent.evaluate`.

    Subclasses must implement :py:meth:`evaluate`.

    Attributes:
        name: Human-readable skill name used in result reports.
        description: One-line description of what the skill measures.
        pass_threshold: Minimum score (inclusive) for ``passed=True``.
            Defaults to ``0.5``.
        weight: Relative weight when aggregating scores across skills.
            Currently informational; aggregation uses equal weights.
    """

    name: str = "base"
    description: str = "Abstract base skill."
    pass_threshold: float = 0.5
    weight: float = 1.0

    @abstractmethod
    def evaluate(self, eval_input: EvalInput) -> SkillResult:
        """Evaluate one aspect of the agent response and return a :class:`SkillResult`.

        Args:
            eval_input: The bundled evaluation input.

        Returns:
            A :class:`~agentevals.models.SkillResult` with ``skill_name`` set to
            ``self.name``.
        """

    def _make_result(
        self,
        score: float,
        reasoning: str = "",
        **details,
    ) -> SkillResult:
        """Convenience factory that builds a :class:`SkillResult` from a score."""
        return SkillResult(
            skill_name=self.name,
            score=round(score, 4),
            passed=score >= self.pass_threshold,
            reasoning=reasoning,
            details=details,
        )
