# CorrectnessSkill

Measures whether the agent's response is **factually correct** relative to a
known reference (ground-truth) answer.

---

## Overview

| Property | Value |
|---|---|
| **Skill name** | `correctness` |
| **Class** | `agentevals.skills.CorrectnessSkill` |
| **Score range** | 0.0 – 1.0 (higher is better) |
| **Default pass threshold** | 0.5 |
| **Required `EvalInput` field** | `reference` |

---

## When to use it

Use `CorrectnessSkill` whenever you have a **ground-truth answer** and want to
verify that the agent reproduces the key facts.  Typical scenarios:

- Question-answering benchmarks with known answers
- Knowledge-retrieval pipelines where the expected answer can be stated precisely
- Regression tests for factual consistency after model updates

If you do **not** have a reference answer, the skill is automatically skipped
and a score of `0.0` is returned with `skipped=True` in the result details.

---

## Constructor

```python
CorrectnessSkill(
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
    pass_threshold: float = 0.5,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_llm` | `bool` | `False` | When `True`, calls an OpenAI chat model instead of using the heuristic. Requires `OPENAI_API_KEY` to be set. |
| `model` | `str` | `"gpt-4o-mini"` | OpenAI model used when `use_llm=True`. Any OpenAI chat-completion model is supported (e.g. `"gpt-4o"`, `"gpt-4-turbo"`). |
| `pass_threshold` | `float` | `0.5` | Minimum score for `SkillResult.passed` to be `True`. Raise this (e.g. `0.8`) for stricter pipelines. |

---

## Evaluation modes

### Heuristic mode (default, `use_llm=False`)

A lightweight, **token F1** comparison between the reference answer and the
response — no API calls, no latency, no cost.

**Algorithm:**

1. Both texts are lowercased and split on non-alphanumeric characters.
2. A set of unique tokens is formed for each text.
3. Precision, recall, and F1 are computed over the token overlap.

```
precision = |overlap| / |response tokens|
recall    = |overlap| / |reference tokens|
F1        = 2 · precision · recall / (precision + recall)
```

The F1 score becomes the `SkillResult.score`.  The `details` dict exposes the
raw `f1` value alongside the score.

**Strengths:** Fast, deterministic, works offline.  
**Limitations:** Ignores word order and semantics; paraphrases may score lower
than they deserve.

### LLM mode (`use_llm=True`)

Sends a structured prompt to an OpenAI model and parses the JSON response.

**Prompt summary:**

> *"Score the response on a scale from 0.0 to 1.0 based on factual correctness
> relative to the reference answer. Respond with `{"score": <float>,
> "reasoning": "<explanation>"}`."*

Scoring rubric used in the prompt:

| Score | Meaning |
|---|---|
| `1.0` | All key facts are correct and complete |
| `0.75` | Mostly correct with minor omissions or imprecisions |
| `0.5` | Partially correct; some important facts are wrong or missing |
| `0.25` | Mostly incorrect; only a small part is right |
| `0.0` | Completely wrong or off-topic |

**Strengths:** Handles paraphrasing, synonyms, and partial answers
intelligently.  
**Limitations:** Requires an API key, adds latency and cost, and introduces
non-determinism.

---

## Return value

`evaluate()` returns a `SkillResult`:

```python
class SkillResult(BaseModel):
    skill_name: str       # "correctness"
    score: float          # 0.0 – 1.0
    passed: bool          # score >= pass_threshold
    reasoning: str        # human-readable explanation
    details: dict         # extra data (see below)
```

### `details` keys

| Key | Present when | Description |
|---|---|---|
| `f1` | Heuristic mode | Raw token F1 score (same as `score`) |
| `skipped` | No reference provided | `True` when the skill was skipped |

---

## Usage examples

### Basic heuristic evaluation

```python
from agentevals import EvalsAgent, EvalInput
from agentevals.skills import CorrectnessSkill

agent = EvalsAgent(skills=[CorrectnessSkill()])

result = agent.evaluate(
    EvalInput(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        reference="Paris is the capital city of France.",
    )
)

skill_result = result.skill("correctness")
print(skill_result.score)      # e.g. 0.8571
print(skill_result.passed)     # True
print(skill_result.reasoning)  # "Token F1=0.86 (precision=0.86, recall=0.86). ..."
```

### LLM-backed evaluation

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

agent = EvalsAgent(
    skills=[CorrectnessSkill(use_llm=True, model="gpt-4o-mini")]
)

result = agent.evaluate(
    EvalInput(
        question="Who wrote Romeo and Juliet?",
        response="Romeo and Juliet was written by William Shakespeare.",
        reference="William Shakespeare wrote Romeo and Juliet.",
    )
)

skill_result = result.skill("correctness")
print(skill_result.score)      # e.g. 1.0
print(skill_result.reasoning)  # LLM explanation
```

### Strict threshold

Raise `pass_threshold` when your pipeline demands high factual accuracy:

```python
agent = EvalsAgent(
    skills=[CorrectnessSkill(pass_threshold=0.8)]
)
```

### Missing reference — skill is skipped

When `EvalInput.reference` is `None` the skill is skipped gracefully:

```python
result = agent.evaluate(
    EvalInput(
        question="What is the capital of France?",
        response="Paris.",
        # reference not provided
    )
)

skill_result = result.skill("correctness")
print(skill_result.score)            # 0.0
print(skill_result.passed)           # False
print(skill_result.details["skipped"])  # True
```

---

## See also

- [`RelevanceSkill`](relevance.md) — measures whether the response addresses the question
- [`FaithfulnessSkill`](faithfulness.md) — measures grounding in a provided context (hallucination detection)
- [`BaseSkill`](../api/base_skill.md) — base class for writing custom skills
- [`EvalInput`](../api/models.md) — full input model reference
