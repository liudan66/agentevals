"""CorrectnessSkill — measures factual correctness against a reference answer."""

from __future__ import annotations

from agentevals.models import EvalInput, SkillResult
from agentevals.skills.base import BaseSkill


_PROMPT_TEMPLATE = """\
You are an expert evaluator. Your task is to assess whether the **Response** is \
factually correct compared to the **Reference Answer**.

Question: {question}
Reference Answer: {reference}
Response to evaluate: {response}

Score the response on a scale from 0.0 to 1.0:
- 1.0  All key facts are correct and complete.
- 0.75 Mostly correct with minor omissions or imprecisions.
- 0.5  Partially correct; some important facts are wrong or missing.
- 0.25 Mostly incorrect; only a small part is right.
- 0.0  Completely wrong or off-topic.

Respond with a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}
"""


class CorrectnessSkill(BaseSkill):
    """Evaluate whether the agent's response is factually correct.

    Requires ``EvalInput.reference`` to be set.  When ``use_llm=True`` an LLM
    performs the evaluation; otherwise a simple keyword-overlap heuristic is used.

    Args:
        use_llm: Use an LLM for evaluation (requires ``OPENAI_API_KEY``).
        model: OpenAI model name (only used when ``use_llm=True``).
        pass_threshold: Minimum score to be considered passing (default ``0.5``).
    """

    name = "correctness"
    description = "Measures factual correctness of the response against a reference answer."

    def __init__(
        self,
        *,
        use_llm: bool = False,
        model: str = "gpt-4o-mini",
        pass_threshold: float = 0.5,
    ) -> None:
        self.use_llm = use_llm
        self.model = model
        self.pass_threshold = pass_threshold

    def evaluate(self, eval_input: EvalInput) -> SkillResult:
        if eval_input.reference is None:
            return self._make_result(
                score=0.0,
                reasoning="No reference answer provided; correctness cannot be evaluated.",
                skipped=True,
            )

        if self.use_llm:
            return self._evaluate_with_llm(eval_input)
        return self._evaluate_heuristic(eval_input)

    def _evaluate_with_llm(self, eval_input: EvalInput) -> SkillResult:
        from agentevals.utils import llm_score

        prompt = _PROMPT_TEMPLATE.format(
            question=eval_input.question,
            reference=eval_input.reference,
            response=eval_input.response,
        )
        result = llm_score(prompt, model=self.model)
        return self._make_result(**result)

    def _evaluate_heuristic(self, eval_input: EvalInput) -> SkillResult:
        """Simple token-overlap heuristic (F1-like score)."""
        ref_tokens = set(_tokenize(eval_input.reference))
        resp_tokens = set(_tokenize(eval_input.response))

        if not ref_tokens:
            return self._make_result(score=0.0, reasoning="Empty reference answer.")

        overlap = ref_tokens & resp_tokens
        precision = len(overlap) / len(resp_tokens) if resp_tokens else 0.0
        recall = len(overlap) / len(ref_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        reasoning = (
            f"Token F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f}). "
            f"Matching tokens: {sorted(overlap)[:10]}"
        )
        return self._make_result(score=f1, reasoning=reasoning, f1=f1)


def _tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumeric characters."""
    import re

    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]
