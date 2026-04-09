"""RelevanceSkill — measures how relevant the response is to the question."""

from __future__ import annotations

from agentevals.models import EvalInput, SkillResult
from agentevals.skills.base import BaseSkill


_PROMPT_TEMPLATE = """\
You are an expert evaluator. Assess whether the **Response** is relevant and \
directly addresses the **Question**.

Question: {question}
Response: {response}

Score from 0.0 to 1.0:
- 1.0  Fully addresses the question with no off-topic content.
- 0.75 Mostly on-topic with minor digressions.
- 0.5  Partially relevant; some key aspects of the question are ignored.
- 0.25 Mostly irrelevant; barely touches the question.
- 0.0  Completely off-topic or refuses to answer.

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}
"""


class RelevanceSkill(BaseSkill):
    """Evaluate whether the agent's response is relevant to the question.

    Args:
        use_llm: Use an LLM for evaluation (requires ``OPENAI_API_KEY``).
        model: OpenAI model name (only used when ``use_llm=True``).
        pass_threshold: Minimum score to be considered passing (default ``0.5``).
    """

    name = "relevance"
    description = "Measures how directly and completely the response addresses the question."

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
        """Heuristic: keyword overlap between question and response."""
        import re

        q_tokens = set(re.split(r"[^a-z0-9]+", eval_input.question.lower()))
        r_tokens = set(re.split(r"[^a-z0-9]+", eval_input.response.lower()))

        # Remove stop words
        stop_words = {
            "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
            "or", "for", "with", "that", "this", "was", "are", "be", "by",
            "at", "as", "do", "did", "has", "have", "had", "not", "but",
            "what", "which", "who", "how", "when", "where", "why", "",
        }
        q_tokens -= stop_words
        r_tokens -= stop_words

        if not q_tokens:
            return self._make_result(
                score=0.5,
                reasoning="Question contains only stop words; relevance assumed neutral.",
            )

        overlap = q_tokens & r_tokens
        score = len(overlap) / len(q_tokens)
        score = min(score, 1.0)

        reasoning = (
            f"Question keywords covered: {len(overlap)}/{len(q_tokens)}. "
            f"Matched: {sorted(overlap)[:10]}"
        )
        return self._make_result(score=score, reasoning=reasoning)
