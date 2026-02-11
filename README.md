# ideanator

A CLI tool that develops vague ideas into well-defined concepts through structured questioning, powered by a local LLM.

## What it does

Most ideas start vague: *"I want to build an app that helps people learn languages."* That sentence is missing who it's for, what problem it solves, why you care, and what makes it different.

**ideanator** runs the **ARISE pipeline** — a 4-phase questioning framework that systematically uncovers what's missing from an idea and asks targeted questions to fill the gaps.

```
$ ideanator

What's your idea? I want to build a language learning app.

  → Scoring vagueness (inverted prompt)...
    Covered: 0/6 | Missing: motivation, audience, problem, vision, risks, differentiation

  ━━ Phase 1 — ANCHOR (Personal Reality) ━━

  It sounds like language learning is something you care about.

  What's been your own experience trying to learn a language,
  and what frustrated you most about the tools out there?

  Are you thinking more about helping complete beginners get started,
  or helping intermediate learners break through a plateau?

Your response: _
```

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/your-username/ideanator.git
cd ideanator
pip install -e ".[dev,mlx]"
```

**Requirements:**
- Python 3.11+
- An MLX-compatible Mac (for local model inference) — or any OpenAI-compatible API server

## Usage

### Interactive mode (default)

You type your idea, then answer questions from the ARISE framework:

```bash
ideanator
```

### Batch mode

Process multiple ideas from a JSON file with simulated user responses (useful for testing prompt efficacy):

```bash
ideanator --file ideas.json --output results.json
```

The input JSON format:

```json
{
  "ideas": [
    {"content": "I want to build an app that helps people learn languages."},
    {"content": "I want to create a budgeting tool for young adults."}
  ]
}
```

### Model selection

```bash
# Use a different MLX model
ideanator --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Point to an already-running server
ideanator --no-server --server-url http://localhost:8080/v1
```

### All options

```
Usage: ideanator [OPTIONS]

Options:
  -f, --file PATH       JSON file with ideas for batch processing
  -m, --model TEXT      Model ID for the MLX server
  -o, --output PATH     Output path for results JSON
  --server-url TEXT     LLM server URL (if already running)
  --no-server           Skip auto-starting the MLX server
  -v, --verbose         Enable verbose debug logging
  --version             Show version
  --help                Show help message
```

## How it works

### The ARISE Framework

ARISE stands for **A**nchor, **R**eveal, **I**magine, **S**cope + **E**valuate. Each phase asks exactly 2 targeted questions in a specific format:

| Phase | Purpose | What it asks |
|-------|---------|-------------|
| **Anchor** | Personal connection | Why do *you* care? What triggered this? |
| **Reveal** | Deeper problem | What's actually broken today? For whom? |
| **Imagine** | Success vision | If this worked perfectly, what would it feel like? |
| **Scope** | Reality check | What could go wrong? What's the smallest test? |

**Anchor and Scope always run.** Reveal and Imagine are conditional — they only run if the idea is missing those dimensions.

### The Inverted Vagueness Scorer

Before questioning begins, the pipeline assesses what's *missing* from the idea across 6 dimensions:

1. **Personal motivation** — Why does the person care?
2. **Target audience** — Who specifically is this for?
3. **Core problem** — What pain or frustration exists today?
4. **Success vision** — What does success look like?
5. **Constraints/risks** — What could go wrong?
6. **Differentiation** — How is this different from alternatives?

The scorer uses an **inverted prompt** — instead of asking "does this idea have X?" (which small models answer YES to due to RLHF sycophancy), it asks "list what is MISSING." Small models are happy to find gaps, which makes the assessment accurate.

**Safety net:** If the model claims nothing is missing but the idea is under 20 words, all dimensions are overridden to missing. Short ideas cannot possibly cover 6 dimensions.

### Anti-Generic Question Check

After each question is generated, it's checked against the original idea's keywords. Questions that share zero meaningful keywords with the idea are flagged as "generic" — they could apply to any idea and aren't specific enough.

## Project Structure

```
ideanator/
├── pyproject.toml              # Package configuration
├── prompts.yaml                # All prompts (externalized for easy iteration)
├── src/ideanator/
│   ├── cli.py                  # Click CLI entry point
│   ├── pipeline.py             # ARISE orchestration loop
│   ├── scorer.py               # Inverted vagueness assessment
│   ├── phases.py               # Phase determination + prompt building
│   ├── parser.py               # Response parsing + anti-generic check
│   ├── llm.py                  # LLM client protocol + MLX server lifecycle
│   ├── prompts.py              # YAML prompt loader
│   ├── types.py                # Data classes and enums
│   └── config.py               # Temperature, token, and model defaults
└── tests/
    ├── conftest.py             # MockLLMClient + shared fixtures
    ├── test_parser.py          # Response parsing + anti-generic tests
    ├── test_scorer.py          # Vagueness scorer + safety net tests
    ├── test_phases.py          # Phase determination + prompt builder tests
    ├── test_prompts.py         # YAML content integrity tests
    ├── test_pipeline.py        # End-to-end pipeline integration tests
    └── test_cli.py             # CLI interface tests
```

### Architecture Decisions

**`LLMClient` Protocol** — Every module that calls the LLM receives an `LLMClient` protocol, never a concrete `OpenAI` instance. This makes the entire pipeline testable with a `MockLLMClient` that returns predetermined responses — no running server needed for tests.

**`prompts.yaml`** — All prompt text is externalized into a YAML file. This means you can iterate on prompts without touching Python code. The file is loaded once and cached.

**Frozen config dataclasses** — Temperature and token configurations use `@dataclass(frozen=True)` to prevent accidental mutation during pipeline execution.

**Phase-based dimension updates** — After each ARISE phase completes, its mapped dimensions are marked as "covered" regardless of the model's actual response. The act of asking about a dimension is what counts, not the quality of the answer.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ideanator

# Run a specific test file
pytest tests/test_parser.py -v
```

## Prompt Customization

Edit `prompts.yaml` to modify any prompt. The file uses Python `str.format()` placeholders:

- `{still_need}` — Comma-separated list of uncovered dimensions
- `{example_user}` — Few-shot example user message
- `{example_response}` — Few-shot example response
- `{conversation}` — Full conversation log
- `{original_idea}` — The user's original idea text

## License

MIT
