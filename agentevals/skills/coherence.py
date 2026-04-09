"""CoherenceSkill — measures the structural and linguistic quality of the response."""

from __future__ import annotations

from agentevals.models import EvalInput, SkillResult
from agentevals.skills.base import BaseSkill


_PROMPT_TEMPLATE = """\
You are an expert evaluator. Assess the **coherence** of the following response — \
clarity of expression, logical flow, and readability.

Question: {question}
Response: {response}

Score from 0.0 to 1.0:
- 1.0  Crystal clear, well-structured, easy to follow.
- 0.75 Mostly clear with minor ambiguities or awkward phrasing.
- 0.5  Understandable but noticeably hard to follow in places.
- 0.25 Confusing in several places; hard to extract meaning.
- 0.0  Incoherent, garbled, or otherwise unreadable.

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}
"""


class CoherenceSkill(BaseSkill):
    """Evaluate the linguistic clarity and logical coherence of a response.

    Args:
        use_llm: Use an LLM for evaluation (requires ``OPENAI_API_KEY``).
        model: OpenAI model name (only used when ``use_llm=True``).
        pass_threshold: Minimum score to be considered passing (default ``0.5``).
    """

    name = "coherence"
    description = "Measures clarity, logical flow, and readability of the response."

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
        """Simple heuristics: sentence count, average length, no empty response."""
        import re

        response = eval_input.response.strip()

        if not response:
            return self._make_result(score=0.0, reasoning="Response is empty.")

        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)

        if num_sentences == 0:
            return self._make_result(score=0.0, reasoning="No complete sentences found.")

        words = response.split()
        avg_sentence_len = len(words) / num_sentences

        # Penalize single very long sentences (>80 words) or very short (<3 words)
        if avg_sentence_len < 3:
            score = 0.4
            reasoning = f"Very short sentences (avg {avg_sentence_len:.1f} words). Possibly incomplete."
        elif avg_sentence_len > 80:
            score = 0.6
            reasoning = f"Very long sentences (avg {avg_sentence_len:.1f} words). May be hard to read."
        else:
            score = 0.8
            reasoning = (
                f"Response has {num_sentences} sentence(s), "
                f"avg length {avg_sentence_len:.1f} words. Appears coherent."
            )

        return self._make_result(
            score=score,
            reasoning=reasoning,
            num_sentences=num_sentences,
            avg_sentence_length=round(avg_sentence_len, 1),
        )
