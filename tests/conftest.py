"""Shared fixtures for agentevals tests."""

from __future__ import annotations

import pytest

from agentevals.models import EvalInput


@pytest.fixture
def basic_input() -> EvalInput:
    return EvalInput(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        reference="Paris is the capital city of France.",
    )


@pytest.fixture
def bad_response_input() -> EvalInput:
    return EvalInput(
        question="What is the capital of France?",
        response="The capital of France is Berlin, a beautiful city in Germany.",
        reference="Paris is the capital city of France.",
    )


@pytest.fixture
def context_input() -> EvalInput:
    return EvalInput(
        question="What is the boiling point of water?",
        response="Water boils at 100 degrees Celsius at sea level.",
        reference="Water boils at 100 °C (212 °F) at standard atmospheric pressure.",
        context=(
            "Water is a chemical compound with the formula H₂O. "
            "Its boiling point is 100 degrees Celsius (212 °F) at sea level (standard atmospheric pressure). "
            "At higher altitudes, the boiling point decreases."
        ),
    )


@pytest.fixture
def no_reference_input() -> EvalInput:
    return EvalInput(
        question="Tell me about Python programming.",
        response="Python is a high-level, interpreted programming language known for its readability.",
    )
