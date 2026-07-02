"""CompletenessSkill — measures whether the response covers all aspects of the question."""

from __future__ import annotations

from agentevals.models import EvalInput, SkillResult
from agentevals.skills.base import BaseSkill


_PROMPT_TEMPLATE = """\
You are an expert evaluator. Assess whether the **Response** fully addresses \
**all aspects** of the **Question**. A complete response leaves no part of the \
question unanswered.

Question: {question}
Response: {response}

Score from 0.0 to 1.0:
- 1.0  Every aspect of the question is fully addressed.
- 0.75 Most aspects covered; 1 minor point overlooked.
- 0.5  Roughly half the question is answered; notable gaps remain.
- 0.25 Only a small portion of the question is addressed.
- 0.0  The response does not address the question at all.

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}
"""


class CompletenessSkill(BaseSkill):
    """Evaluate whether the agent's response fully covers all aspects of the question.

    Unlike :class:`~agentevals.skills.relevance.RelevanceSkill`, which checks
    *whether* the response is on-topic, this skill checks *how completely* all
    parts of the question are answered.

    When ``use_llm=True`` an LLM performs the evaluation; otherwise a
    heuristic based on sub-question detection and keyword coverage is used.

    Args:
        use_llm: Use an LLM for evaluation (requires ``OPENAI_API_KEY``).
        model: OpenAI model name (only used when ``use_llm=True``).
        pass_threshold: Minimum score to be considered passing (default ``0.5``).
    """

    name = "completeness"
    description = "Measures whether the response fully addresses all aspects of the question."

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
        """Heuristic: detect sub-questions in the question, then check coverage.

        Strategy
        --------
        1. Split the question on conjunctions and punctuation that typically
           separate multiple requests (``and``, ``also``, ``?``, ``;``).
        2. For each sub-question extract its meaningful (non-stop-word) keywords.
        3. A sub-question is considered *covered* when at least half of its
           keywords appear in the response.
        4. The score is the fraction of sub-questions covered.
        """
        import re

        response = eval_input.response.strip()
        if not response:
            return self._make_result(
                score=0.0,
                reasoning="Response is empty.",
                sub_questions_total=0,
                sub_questions_covered=0,
            )

        # Split question into sub-questions
        raw_parts = re.split(r"[?;]|(?:\band\b)|(?:\balso\b)", eval_input.question, flags=re.IGNORECASE)
        sub_questions = [p.strip() for p in raw_parts if p.strip()]

        if not sub_questions:
            return self._make_result(
                score=0.5,
                reasoning="Could not identify distinct sub-questions; completeness assumed neutral.",
                sub_questions_total=0,
                sub_questions_covered=0,
            )

        stop_words = {
            "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
            "or", "for", "with", "that", "this", "was", "are", "be", "by",
            "at", "as", "do", "did", "has", "have", "had", "not", "but",
            "what", "which", "who", "how", "when", "where", "why", "tell",
            "me", "you", "your", "i", "my", "please", "",
        }

        def keywords(text: str) -> set[str]:
            tokens = re.split(r"[^a-z0-9]+", text.lower())
            return {t for t in tokens if t and t not in stop_words}

        resp_kw = keywords(response)
        covered = 0
        coverage_detail: list[str] = []

        for sq in sub_questions:
            sq_kw = keywords(sq)
            if not sq_kw:
                covered += 1  # vacuously covered
                coverage_detail.append(f"'{sq}' → (no keywords, assumed covered)")
                continue
            overlap = sq_kw & resp_kw
            ratio = len(overlap) / len(sq_kw)
            is_covered = ratio >= 0.5
            if is_covered:
                covered += 1
            coverage_detail.append(
                f"'{sq}' → {len(overlap)}/{len(sq_kw)} keywords matched "
                f"({'✓' if is_covered else '✗'})"
            )

        total = len(sub_questions)
        score = covered / total if total else 0.0

        reasoning = (
            f"{covered}/{total} sub-question(s) covered. "
            + " | ".join(coverage_detail)
        )
        return self._make_result(
            score=score,
            reasoning=reasoning,
            sub_questions_total=total,
            sub_questions_covered=covered,
        )
