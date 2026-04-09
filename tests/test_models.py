"""Tests for data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentevals.models import EvalInput, EvalResult, SkillResult


class TestEvalInput:
    def test_required_fields(self):
        inp = EvalInput(question="q", response="r")
        assert inp.question == "q"
        assert inp.response == "r"
        assert inp.reference is None
        assert inp.context is None
        assert inp.metadata == {}

    def test_optional_fields(self):
        inp = EvalInput(
            question="q",
            response="r",
            reference="ref",
            context="ctx",
            metadata={"key": "value"},
        )
        assert inp.reference == "ref"
        assert inp.context == "ctx"
        assert inp.metadata["key"] == "value"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            EvalInput(question="q")  # missing response


class TestSkillResult:
    def test_valid(self):
        sr = SkillResult(skill_name="test", score=0.8, passed=True)
        assert sr.skill_name == "test"
        assert sr.score == 0.8
        assert sr.passed is True

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SkillResult(skill_name="test", score=1.5, passed=True)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            SkillResult(skill_name="test", score=-0.1, passed=False)


class TestEvalResult:
    def test_empty_skill_results(self):
        result = EvalResult()
        assert result.overall_score == 0.0
        assert result.passed is False

    def test_aggregation(self):
        sr1 = SkillResult(skill_name="a", score=0.8, passed=True)
        sr2 = SkillResult(skill_name="b", score=0.6, passed=True)
        result = EvalResult(skill_results=[sr1, sr2])
        assert result.overall_score == pytest.approx(0.7, abs=1e-3)
        assert result.passed is True

    def test_fails_if_any_skill_fails(self):
        sr1 = SkillResult(skill_name="a", score=0.9, passed=True)
        sr2 = SkillResult(skill_name="b", score=0.3, passed=False)
        result = EvalResult(skill_results=[sr1, sr2])
        assert result.passed is False

    def test_skill_lookup(self):
        sr = SkillResult(skill_name="coherence", score=0.75, passed=True)
        result = EvalResult(skill_results=[sr])
        found = result.skill("coherence")
        assert found is not None
        assert found.score == 0.75

    def test_skill_lookup_missing_returns_none(self):
        result = EvalResult()
        assert result.skill("nonexistent") is None
