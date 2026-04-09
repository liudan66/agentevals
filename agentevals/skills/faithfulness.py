"""FaithfulnessSkill — detects hallucinations / grounding issues."""

from __future__ import annotations

from agentevals.models import EvalInput, SkillResult
from agentevals.skills.base import BaseSkill


_PROMPT_TEMPLATE = """\
You are an expert fact-checker. Determine whether all claims in the **Response** \
are supported by the **Context** provided. Any claim that cannot be verified from \
the context is a hallucination.

Context: {context}
Question: {question}
Response: {response}

Score from 0.0 to 1.0:
- 1.0  Every claim in the response is grounded in the context.
- 0.75 Most claims are grounded; 1–2 minor unsupported statements.
- 0.5  Some claims are grounded but there are notable hallucinations.
- 0.25 Most claims are hallucinated or cannot be verified from the context.
- 0.0  The response contradicts or completely ignores the context.

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}
"""


class FaithfulnessSkill(BaseSkill):
    """Evaluate whether the agent's response is grounded in the provided context.

    Requires ``EvalInput.context`` to be set.  When no context is supplied the
    skill is skipped and a neutral score is returned.

    Args:
        use_llm: Use an LLM for evaluation (requires ``OPENAI_API_KEY``).
        model: OpenAI model name (only used when ``use_llm=True``).
        pass_threshold: Minimum score to be considered passing (default ``0.5``).
    """

    name = "faithfulness"
    description = "Measures whether the response is grounded in the provided context (no hallucinations)."

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
        if eval_input.context is None:
            return self._make_result(
                score=1.0,
                reasoning="No context provided; faithfulness check skipped.",
                skipped=True,
            )

        if self.use_llm:
            return self._evaluate_with_llm(eval_input)
        return self._evaluate_heuristic(eval_input)

    def _evaluate_with_llm(self, eval_input: EvalInput) -> SkillResult:
        from agentevals.utils import llm_score

        prompt = _PROMPT_TEMPLATE.format(
            context=eval_input.context,
            question=eval_input.question,
            response=eval_input.response,
        )
        result = llm_score(prompt, model=self.model)
        return self._make_result(**result)

    def _evaluate_heuristic(self, eval_input: EvalInput) -> SkillResult:
        """Heuristic: fraction of response tokens that appear in the context."""
        import re

        def tokenize(text: str) -> set[str]:
            stop = {
                "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
                "or", "for", "with", "that", "this", "was", "are", "be", "by",
                "at", "as", "do", "did", "has", "have", "had", "not", "but", "",
            }
            return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t} - stop

        ctx_tokens = tokenize(eval_input.context)
        resp_tokens = tokenize(eval_input.response)

        if not resp_tokens:
            return self._make_result(
                score=0.0,
                reasoning="Response is empty.",
            )

        overlap = resp_tokens & ctx_tokens
        score = len(overlap) / len(resp_tokens)

        reasoning = (
            f"{len(overlap)}/{len(resp_tokens)} response tokens found in context "
            f"(faithfulness={score:.2f})."
        )
        return self._make_result(score=score, reasoning=reasoning)
