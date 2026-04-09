"""ConcisenessSkill — measures whether the response is appropriately concise."""

from __future__ import annotations

from agentevals.models import EvalInput, SkillResult
from agentevals.skills.base import BaseSkill


_PROMPT_TEMPLATE = """\
You are an expert evaluator. Assess whether the **Response** is appropriately \
concise for the given **Question** — it should answer the question fully without \
unnecessary repetition, padding, or verbosity.

Question: {question}
Response: {response}

Score from 0.0 to 1.0:
- 1.0  Perfectly concise: complete answer, no fluff.
- 0.75 Slightly verbose but still focused.
- 0.5  Noticeably wordy or contains some padding.
- 0.25 Very verbose; key answer buried in unnecessary text.
- 0.0  Extremely padded, repetitive, or the response gives a non-answer.

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}
"""

# Responses beyond this many words are considered verbose; below this is ideal.
_VERBOSE_THRESHOLD = 300
_IDEAL_MAX = 150


class ConcisenessSkill(BaseSkill):
    """Evaluate whether the response is concise.

    Args:
        use_llm: Use an LLM for evaluation (requires ``OPENAI_API_KEY``).
        model: OpenAI model name (only used when ``use_llm=True``).
        pass_threshold: Minimum score to be considered passing (default ``0.5``).
        verbose_threshold: Word count above which the score is penalized.
        ideal_max: Word count below which the response is rewarded.
    """

    name = "conciseness"
    description = "Measures whether the response is concise without losing completeness."

    def __init__(
        self,
        *,
        use_llm: bool = False,
        model: str = "gpt-4o-mini",
        pass_threshold: float = 0.5,
        verbose_threshold: int = _VERBOSE_THRESHOLD,
        ideal_max: int = _IDEAL_MAX,
    ) -> None:
        self.use_llm = use_llm
        self.model = model
        self.pass_threshold = pass_threshold
        self.verbose_threshold = verbose_threshold
        self.ideal_max = ideal_max

    def evaluate(self, eval_input: EvalInput) -> SkillResult:
        if self.use_llm:
            return self._evaluate_with_llm(eval_input)
        return self._evaluate_heuristic(eval_input)

    def _evaluate_with_llm(self, eval_input: EvalInput) -> SkillResult:
        from agentevals.utils import llm_score

        prompt = _PROMPT_TEMPLATE.format(
            question=eval_input.question,
            response=eval_input.response,
        )
        result = llm_score(prompt, model=self.model)
        return self._make_result(**result)

    def _evaluate_heuristic(self, eval_input: EvalInput) -> SkillResult:
        """Penalize responses that exceed the verbosity threshold."""
        word_count = len(eval_input.response.split())

        if word_count == 0:
            return self._make_result(score=0.0, reasoning="Response is empty.", word_count=0)

        if word_count <= self.ideal_max:
            score = 1.0
            reasoning = f"Response is concise ({word_count} words)."
        elif word_count <= self.verbose_threshold:
            # Linear decay from 1.0 to 0.5 between ideal_max and verbose_threshold
            ratio = (word_count - self.ideal_max) / (self.verbose_threshold - self.ideal_max)
            score = 1.0 - ratio * 0.5
            reasoning = f"Response is somewhat verbose ({word_count} words)."
        else:
            score = 0.4
            reasoning = f"Response is very verbose ({word_count} words > threshold {self.verbose_threshold})."

        return self._make_result(score=score, reasoning=reasoning, word_count=word_count)
