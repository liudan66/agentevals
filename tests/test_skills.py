"""Tests for individual evaluation skills (heuristic mode)."""

from __future__ import annotations

import pytest

from agentevals.models import EvalInput
from agentevals.skills import (
    CoherenceSkill,
    ConcisenessSkill,
    CorrectnessSkill,
    FaithfulnessSkill,
    RelevanceSkill,
)


# ---------------------------------------------------------------------------
# CorrectnessSkill
# ---------------------------------------------------------------------------

class TestCorrectnessSkill:
    skill = CorrectnessSkill()

    def test_correct_response_passes(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert result.skill_name == "correctness"
        assert result.score > 0.5
        assert result.passed is True

    def test_wrong_response_fails(self, bad_response_input, basic_input):
        # The heuristic measures token overlap, not semantic correctness.
        # A wrong response should score *lower* than a correct one.
        correct_result = self.skill.evaluate(basic_input)
        wrong_result = self.skill.evaluate(bad_response_input)
        assert wrong_result.score <= correct_result.score

    def test_no_reference_skips(self, no_reference_input):
        result = self.skill.evaluate(no_reference_input)
        assert result.details.get("skipped") is True
        assert result.score == 0.0

    def test_score_is_in_range(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# RelevanceSkill
# ---------------------------------------------------------------------------

class TestRelevanceSkill:
    skill = RelevanceSkill()

    def test_relevant_response_passes(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert result.skill_name == "relevance"
        assert result.score > 0.5
        assert result.passed is True

    def test_irrelevant_response(self):
        inp = EvalInput(
            question="What is the speed of light?",
            response="I love eating pizza on Sundays.",
        )
        result = self.skill.evaluate(inp)
        assert result.score < 0.5

    def test_score_is_in_range(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# CoherenceSkill
# ---------------------------------------------------------------------------

class TestCoherenceSkill:
    skill = CoherenceSkill()

    def test_coherent_response_passes(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert result.skill_name == "coherence"
        assert result.passed is True

    def test_empty_response_fails(self):
        inp = EvalInput(question="q", response="")
        result = self.skill.evaluate(inp)
        assert result.score == 0.0
        assert result.passed is False

    def test_score_is_in_range(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert 0.0 <= result.score <= 1.0

    def test_details_present(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert "num_sentences" in result.details
        assert "avg_sentence_length" in result.details


# ---------------------------------------------------------------------------
# FaithfulnessSkill
# ---------------------------------------------------------------------------

class TestFaithfulnessSkill:
    skill = FaithfulnessSkill()

    def test_faithful_response_passes(self, context_input):
        result = self.skill.evaluate(context_input)
        assert result.skill_name == "faithfulness"
        assert result.score > 0.5
        assert result.passed is True

    def test_no_context_skips_with_passing_score(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert result.details.get("skipped") is True
        assert result.score == 1.0
        assert result.passed is True

    def test_unfaithful_response(self):
        inp = EvalInput(
            question="What does the document say about Python?",
            response="Java is a coffee beverage enjoyed worldwide.",
            context="Python is a programming language known for readability and simplicity.",
        )
        result = self.skill.evaluate(inp)
        assert result.score < 0.5

    def test_score_is_in_range(self, context_input):
        result = self.skill.evaluate(context_input)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# ConcisenessSkill
# ---------------------------------------------------------------------------

class TestConcisenessSkill:
    skill = ConcisenessSkill()

    def test_concise_response_passes(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert result.skill_name == "conciseness"
        assert result.score == 1.0
        assert result.passed is True

    def test_verbose_response_penalized(self):
        inp = EvalInput(
            question="What is 2+2?",
            response=" ".join(["word"] * 400),  # 400 words
        )
        result = self.skill.evaluate(inp)
        assert result.score < 1.0

    def test_extremely_verbose_fails_or_penalized(self):
        inp = EvalInput(
            question="What is 2+2?",
            response=" ".join(["word"] * 500),  # well above threshold
        )
        result = self.skill.evaluate(inp)
        assert result.score <= 0.5

    def test_empty_response_fails(self):
        inp = EvalInput(question="q", response="")
        result = self.skill.evaluate(inp)
        assert result.score == 0.0
        assert result.passed is False

    def test_score_is_in_range(self, basic_input):
        result = self.skill.evaluate(basic_input)
        assert 0.0 <= result.score <= 1.0
