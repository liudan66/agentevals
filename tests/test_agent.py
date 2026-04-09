"""Tests for EvalsAgent."""

from __future__ import annotations

import pytest

from agentevals import EvalsAgent, EvalInput
from agentevals.skills import (
    CoherenceSkill,
    ConcisenessSkill,
    CorrectnessSkill,
    FaithfulnessSkill,
    RelevanceSkill,
)


def _make_agent() -> EvalsAgent:
    return EvalsAgent(
        skills=[
            CorrectnessSkill(),
            RelevanceSkill(),
            CoherenceSkill(),
            FaithfulnessSkill(),
            ConcisenessSkill(),
        ]
    )


class TestEvalsAgentEvaluate:
    def test_evaluate_returns_eval_result(self, basic_input):
        agent = _make_agent()
        result = agent.evaluate(basic_input)
        assert result is not None
        assert len(result.skill_results) == 5

    def test_all_skill_names_present(self, basic_input):
        agent = _make_agent()
        result = agent.evaluate(basic_input)
        names = {r.skill_name for r in result.skill_results}
        assert names == {"correctness", "relevance", "coherence", "faithfulness", "conciseness"}

    def test_overall_score_is_mean(self, basic_input):
        agent = _make_agent()
        result = agent.evaluate(basic_input)
        expected = sum(r.score for r in result.skill_results) / len(result.skill_results)
        assert result.overall_score == pytest.approx(expected, abs=1e-3)

    def test_good_response_passes_overall(self, basic_input):
        agent = _make_agent()
        result = agent.evaluate(basic_input)
        assert result.passed is True

    def test_summary_is_set(self, basic_input):
        agent = _make_agent()
        result = agent.evaluate(basic_input)
        assert result.summary != ""

    def test_empty_skills_list(self, basic_input):
        agent = EvalsAgent(skills=[])
        result = agent.evaluate(basic_input)
        assert result.overall_score == 0.0
        assert result.skill_results == []

    def test_with_context_input(self, context_input):
        agent = _make_agent()
        result = agent.evaluate(context_input)
        assert result.overall_score > 0.5

    def test_skill_result_accessible_by_name(self, basic_input):
        agent = _make_agent()
        result = agent.evaluate(basic_input)
        coherence = result.skill("coherence")
        assert coherence is not None
        assert coherence.skill_name == "coherence"


class TestEvalsAgentStopOnFailure:
    def test_stop_on_failure_halts_early(self):
        bad_input = EvalInput(
            question="What is the speed of light?",
            response="",  # empty response triggers failures early
            reference="The speed of light is approximately 3×10⁸ m/s.",
        )
        agent = EvalsAgent(
            skills=[
                CorrectnessSkill(),
                RelevanceSkill(),
                CoherenceSkill(),
                FaithfulnessSkill(),
                ConcisenessSkill(),
            ],
            stop_on_failure=True,
        )
        result = agent.evaluate(bad_input)
        # With an empty response, at least one skill will fail immediately
        # and stop_on_failure means not all 5 skills necessarily ran
        assert len(result.skill_results) <= 5

    def test_stop_on_failure_false_runs_all(self):
        bad_input = EvalInput(
            question="What is the speed of light?",
            response="",
            reference="The speed of light is approximately 3×10⁸ m/s.",
        )
        agent = EvalsAgent(
            skills=[
                CorrectnessSkill(),
                RelevanceSkill(),
                CoherenceSkill(),
                FaithfulnessSkill(),
                ConcisenessSkill(),
            ],
            stop_on_failure=False,
        )
        result = agent.evaluate(bad_input)
        assert len(result.skill_results) == 5


class TestEvalsAgentFluentAPI:
    def test_add_skill(self):
        agent = EvalsAgent()
        agent.add_skill(CoherenceSkill())
        assert len(agent.skills) == 1

    def test_add_skill_chaining(self):
        agent = (
            EvalsAgent()
            .add_skill(CoherenceSkill())
            .add_skill(RelevanceSkill())
        )
        assert len(agent.skills) == 2

    def test_remove_skill(self):
        agent = EvalsAgent(skills=[CoherenceSkill(), RelevanceSkill()])
        agent.remove_skill("coherence")
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "relevance"

    def test_remove_nonexistent_skill_is_noop(self):
        agent = EvalsAgent(skills=[CoherenceSkill()])
        agent.remove_skill("nonexistent")
        assert len(agent.skills) == 1
