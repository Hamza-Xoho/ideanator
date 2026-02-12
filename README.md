# ideanator

A cross-platform CLI tool that develops vague ideas into well-defined concepts through structured questioning, powered by a local LLM.

Works with **Ollama** (Linux, macOS, Windows), **MLX** (macOS + Apple Silicon), or any OpenAI-compatible API server.

---

### Contents

- [What it does](#what-it-does)
- [Installation](#installation)
  - [Quick install](#quick-install-recommended)
  - [Updating](#updating)
  - [Uninstalling](#uninstalling)
- [Recommended Models](#recommended-models)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Interactive mode](#interactive-mode-default)
  - [Batch mode](#batch-mode)
  - [Model selection](#model-selection)
  - [All options](#all-options)
  - [Backend defaults](#backend-defaults)
- [How it works](#how-it-works)
  - [The ARISE Framework](#the-arise-framework)
  - [The Inverted Vagueness Scorer](#the-inverted-vagueness-scorer)
  - [Anti-Generic Question Check](#anti-generic-question-check)
  - [Three-Stage Refactoring Engine](#three-stage-refactoring-engine)
- [Prompt Customization](#prompt-customization)
- [Running Tests](#running-tests)
- [Architecture](#architecture)
  - [Project structure](#project-structure)
  - [Data flow](#data-flow)
  - [Module reference](#module-reference)
  - [Design patterns](#design-patterns)
  - [Configuration system](#configuration-system)
  - [LLM abstraction layer](#llm-abstraction-layer)
  - [Server lifecycle management](#server-lifecycle-management)
  - [Pipeline internals](#pipeline-internals)
  - [Testing strategy](#testing-strategy)
  - [Adding a new backend](#adding-a-new-backend)
- [License](#license)

---

## What it does

Most ideas start vague: *"I want to build an app that helps people learn languages."* That sentence is missing who it's for, what problem it solves, why you care, and what makes it different.

**ideanator** runs the **ARISE pipeline** — a 4-phase questioning framework that systematically uncovers what's missing from an idea and asks targeted questions to fill the gaps. After the conversation, a **three-stage refactoring engine** (Extract, Synthesize, Validate) transforms the raw Q&A into a structured, faithful, non-sycophantic idea statement.

```
$ ideanator --ollama

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

After all phases complete, the refactoring engine produces a structured output:

```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  REFINED IDEA STATEMENT
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ONE-LINER: I'm building a conversational Spanish practice tool
  for college students who find Duolingo too gamified and not
  focused enough on real dialogue.

  PROBLEM: College students trying to learn conversational Spanish
  hit a wall — existing apps drill vocabulary but never simulate
  actual conversations. Students spend months on apps and still
  can't hold a 5-minute chat.

  SOLUTION: A practice tool that pairs learners with AI-driven
  conversation partners calibrated to their level, focusing
  purely on spoken dialogue rather than grammar drills.

  AUDIENCE: College students taking Spanish courses who want to
  supplement classroom learning with real conversation practice.

  DIFFERENTIATOR: Unlike Duolingo and Babbel which sort by
  grammar level, this matches on conversational ability and
  focuses exclusively on dialogue — no flashcards, no points.

  OPEN QUESTIONS:
    • How would you measure conversational improvement?
    • What's the minimum viable conversation scenario?
    • How do you handle different Spanish dialects?

  EXPLORATION STATUS:
    Problem: ✅ Well-explored
    Audience: ✅ Well-explored
    Solution: ⚠️ Partially explored
    Differentiation: ✅ Well-explored
    Motivation: ✅ Well-explored

  VALIDATION: confidence=0.87 | refinement rounds=0
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Installation

### Quick install (recommended)

Install directly from GitHub — no git clone needed:

```bash
# Using pipx (recommended — installs in an isolated environment)
pipx install git+https://github.com/Hamza-Xoho/ideanatorCLI.git

# Using pip
pip install git+https://github.com/Hamza-Xoho/ideanatorCLI.git
```

### With MLX support (macOS Apple Silicon)

```bash
pip install "ideanator[mlx] @ git+https://github.com/Hamza-Xoho/ideanatorCLI.git"
```

### Development install

```bash
git clone https://github.com/Hamza-Xoho/ideanatorCLI.git
cd ideanatorCLI
make install-dev    # or: pip install -e ".[dev]"
```

### Updating

```bash
# pipx
pipx upgrade ideanator

# pip
pip install --upgrade git+https://github.com/Hamza-Xoho/ideanatorCLI.git

# Development (from cloned repo)
make update         # or: git pull origin main && pip install .
```

### Uninstalling

```bash
# pipx
pipx uninstall ideanator

# pip
pip uninstall ideanator

# Development (from cloned repo)
make uninstall
```

**Requirements:**
- Python 3.11+
- One of the following LLM backends:

| Backend | Platform | Install |
|---------|----------|---------|
| **Ollama** | Linux, macOS, Windows | [ollama.com](https://ollama.com) |
| **MLX** | macOS (Apple Silicon) | See "With MLX support" above |
| **External** | Any | Any OpenAI-compatible server |

---

## Recommended Models

I personally recommend using **Ollama** with either **`qwen2.5:7b-instruct`** or **`llama3.2:latest`**. Both ran great on my M2 MacBook Air with 16GB RAM.

### Setting up Ollama with the recommended models

**1. Install Ollama:**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com for macOS / Windows
```

**2. Pull your preferred model:**

```bash
# Option A: Qwen 2.5 7B Instruct (best quality, ~4.7GB download)
ollama pull qwen2.5:7b-instruct

# Option B: Llama 3.2 (good balance of speed and quality, ~2GB download)
ollama pull llama3.2:latest
```

**3. Run ideanator with your model:**

```bash
# With Qwen 2.5
ideanator --ollama -m qwen2.5:7b-instruct

# With Llama 3.2
ideanator --ollama -m llama3.2:latest
```

Both models handle the structured output formats well and produce specific, non-generic questions. The 7B Qwen model gives slightly richer responses; the 3B Llama model is faster. Either works well for the full pipeline including the three-stage refactoring engine.

> **Note:** ideanator handles starting Ollama and pulling models automatically. If you just run `ideanator --ollama -m qwen2.5:7b-instruct`, it will start the Ollama daemon and pull the model if needed.

---

## Quick Start

### With Ollama (recommended)

```bash
# Install Ollama from https://ollama.com, then:
ideanator --ollama -m qwen2.5:7b-instruct
```

ideanator will automatically start the Ollama daemon, pull the model, and begin the interactive session.

### With MLX (macOS + Apple Silicon)

```bash
pip install "ideanator[mlx] @ git+https://github.com/Hamza-Xoho/ideanatorCLI.git"
ideanator --mlx
```

### With any running server

```bash
ideanator --external --server-url http://localhost:1234/v1
```

---

## Usage

### Interactive mode (default)

You type your idea, then answer questions from the ARISE framework:

```bash
ideanator --ollama
ideanator --ollama -m qwen2.5:7b-instruct
ideanator --external --server-url http://localhost:1234/v1
```

### Batch mode

Process multiple ideas from a JSON file with simulated user responses. Useful for testing prompt efficacy at scale:

```bash
ideanator --ollama -f ideas.json -o results.json
ideanator --ollama -m llama3.2:latest -f ideas.json
```

Input JSON format:

```json
{
  "ideas": [
    {"content": "I want to build an app that helps people learn languages."},
    {"content": "I want to create a budgeting tool for young adults."}
  ]
}
```

Results are saved incrementally — if the process is interrupted, you keep everything that completed.

### Model selection

```bash
# Ollama with recommended models
ideanator --ollama -m qwen2.5:7b-instruct
ideanator --ollama -m llama3.2:latest

# MLX with a specific model
ideanator --mlx -m mlx-community/Llama-3.2-1B-Instruct-4bit

# External server with a custom model name
ideanator --external -m my-model --server-url http://localhost:9999/v1
```

### All options

```
BACKENDS (pick one)
  --ollama            Use Ollama  (Linux, macOS, Windows)
  --mlx               Use MLX     (macOS + Apple Silicon)
  --external          Use any already-running server
  (default: --ollama)

OPTIONS
  -m, --model ID      Model to use (default depends on backend)
  --server-url URL    Override the server URL
  -f, --file PATH     Batch mode: process ideas from a JSON file
  -o, --output PATH   Save results to a JSON file
  -v, --verbose       Show debug logs
  --version           Show version
  --help              Show this help
```

### Backend defaults

Each backend has sensible defaults so you don't need to memorize model IDs or ports:

| Backend | Default model | Default URL |
|---------|--------------|-------------|
| `--ollama` | `llama3.2:3b` | `http://localhost:11434/v1` |
| `--mlx` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `http://localhost:8080/v1` |
| `--external` | `default` | `http://localhost:8080/v1` |

Override any default with `-m` or `--server-url`.

---

## How it works

### The ARISE Framework

ARISE stands for **A**nchor, **R**eveal, **I**magine, **S**cope + **E**valuate. Each phase asks exactly 2 targeted questions in a structured format:

| Phase | Purpose | What it asks |
|-------|---------|-------------|
| **Anchor** | Personal connection | Why do *you* care? What triggered this? |
| **Reveal** | Deeper problem | What's actually broken today? For whom? |
| **Imagine** | Success vision | If this worked perfectly, what would it feel like? |
| **Scope** | Reality check | What could go wrong? What's the smallest test? |

**Anchor and Scope always run.** Reveal and Imagine are conditional — they only activate if the idea is missing those dimensions (determined by the vagueness scorer).

### The Inverted Vagueness Scorer

Before questioning begins, the pipeline assesses what's *missing* from the idea across 6 dimensions:

1. **Personal motivation** — Why does the person care?
2. **Target audience** — Who specifically is this for?
3. **Core problem** — What pain or frustration exists today?
4. **Success vision** — What does success look like?
5. **Constraints/risks** — What could go wrong?
6. **Differentiation** — How is this different from alternatives?

The scorer uses an **inverted prompt** — instead of asking "does this idea have X?" (which small models answer YES to due to RLHF sycophancy), it asks "list what is MISSING." Small models are happy to find gaps, which makes the assessment accurate even on 3B parameter models.

**Safety net:** If the model claims nothing is missing but the idea is under 20 words, all dimensions are overridden to missing. A 12-word sentence cannot possibly cover 6 distinct dimensions — this catches false negatives from overly agreeable models.

### Anti-Generic Question Check

After each question is generated, it's checked against the original idea's keywords. Keywords are words with 4+ characters that aren't in a curated stop-word list (e.g., "want", "build", "create", "people"). If a question shares zero meaningful keywords with the idea, it's flagged as "generic" — it could apply to any idea and isn't specific enough.

This is a fast heuristic check (word overlap, not ML-based), so it adds no latency and is fully explainable.

### Three-Stage Refactoring Engine

After the ARISE questioning phases complete, the full conversation is passed through a three-stage LLM pipeline that transforms the raw Q&A into a structured idea statement. This follows research showing that multiple focused LLM calls produce significantly better results than a single monolithic prompt.

```
Q&A Transcript
     │
     ▼
STAGE 1: EXTRACT (temp=0.3, JSON output)
  Parse → structured dimensions + user quotes + gaps + contradictions
     │
     ▼
STAGE 2: SYNTHESIZE (temp=0.5, text output)
  Chain-of-density: iteratively densify the idea statement
  Banned words enforced, user's exact phrases preserved
     │
     ▼
STAGE 3: VALIDATE (temp=0.2, JSON output)
  Faithfulness + completeness + sycophancy checks
     │
     ├── confidence ≥ 0.8 → Output final statement
     └── confidence < 0.8 → Self-refine loop back to Stage 2 (max 2 rounds)
```

**Stage 1 — Extract** parses the conversation into structured dimensions: problem, audience, solution, differentiation, motivation, the user's verbatim key phrases, any contradictions between earlier and later statements, and unresolved questions. Every claim cites its conversation turn for auditability.

**Stage 2 — Synthesize** uses chain-of-density adapted prompting to iteratively build specificity. It enforces a banned-words list of 44 phrases (vague intensifiers like "innovative" and "robust," corporate filler like "leverage" and "synergy," and LLM tells like "delve" and "landscape"). The output is written in first person to maximize the user's sense of ownership, and matches the user's language register (casual/formal/technical).

**Stage 3 — Validate** performs three checks:
- **Faithfulness** — Is each sentence supported by the transcript, or hallucinated?
- **Completeness** — Are the four must-have dimensions (problem, audience, solution, differentiation) covered?
- **Sycophancy** — Did the refinement make the idea sound better than the user actually described?

If confidence falls below 0.8, the critique is fed back to Stage 2 for revision (up to 2 rounds).

**Output format** — The refactored statement has six sections (150-250 words total):

| Section | What it covers |
|---------|---------------|
| **One-liner** | Under 30 words: For [who] who [problem], [solution] that [differentiator] |
| **Problem** | 2-3 sentences: specific pain point with quantified impact |
| **Solution** | 2-3 sentences: what the idea does and the unique approach |
| **Audience** | 1-2 sentences: specific persona, not "everyone" |
| **Differentiator** | 1-2 sentences: comparison to named alternatives |
| **Open questions** | Bulleted gaps that still need exploration |

**Exploration status** is computed programmatically from the conversation structure (not by asking the LLM to self-assess). It counts how many words the user provided per dimension and labels each as:
- ✅ **Well-explored** — Phase ran and user gave a substantive response (20+ words)
- ⚠️ **Partially explored** — Phase ran but user response was thin
- 🔲 **Not yet explored** — The relevant phase didn't run

**Contradiction detection** checks for negation patterns across user responses in different phases (e.g., saying "not about cost" in one phase but discussing pricing in another). Contradictions from both programmatic detection and the LLM extraction stage are merged and surfaced.

---

## Prompt Customization

Prompts are bundled inside the package at `src/ideanator/data/`:

**`data/prompts.yaml`** — The ARISE questioning prompts, simulated user prompt, legacy synthesis prompt, and few-shot example pool. Edit this file to modify questioning behavior without touching code. Uses Python `str.format()` placeholders:

| Placeholder | Description | Used in |
|-------------|-------------|---------|
| `{still_need}` | Comma-separated list of uncovered dimensions | Phase prompts |
| `{example_user}` | Few-shot example user message | Phase prompts |
| `{example_response}` | Few-shot example assistant response | Phase prompts |
| `{conversation}` | Full conversation log so far | Reveal, Imagine, Scope, Synthesis |
| `{original_idea}` | The user's original idea text | Simulated user prompt |

**`data/prompts/`** directory — The three-stage refactoring engine configs. Each YAML file specifies the system prompt, user template, model settings (temperature, max_tokens), and anti-patterns:

| File | Stage | Temperature | Tokens | Purpose |
|------|-------|-------------|--------|---------|
| `data/prompts/extract.yml` | Extract | 0.3 | 800 | Parse conversation into structured dimensions |
| `data/prompts/synthesize.yml` | Synthesize | 0.5 | 600 | Chain-of-density synthesis with banned words |
| `data/prompts/validate.yml` | Validate | 0.2 | 600 | Faithfulness, completeness, sycophancy checks |

The `synthesize.yml` file contains the banned-words list under `anti_patterns.banned_phrases`. Add or remove phrases without touching any code.

Each phase has 2-3 few-shot examples in the `example_pool` section of `prompts.yaml`. A random example is selected each run to prevent deterministic output.

---

## Running Tests

All 205 tests run without a server — the entire pipeline is tested through `MockLLMClient`.

```bash
# Using Make (recommended)
make test           # Run all tests
make test-cov       # Run with coverage report
make lint           # Run ruff linter
make typecheck      # Run mypy

# Or directly
pytest tests/ -v
pytest --cov=ideanator

# Run a specific test file
pytest tests/test_refactor.py -v

# Run a specific test class
pytest tests/test_pipeline.py::TestRunAriseForIdea -v
```

---

## Architecture

> This section is for developers and ML/AI engineers who want to understand, extend, or contribute to the codebase.

### Project structure

```
ideanatorCLI/
├── pyproject.toml                    # Build config, deps, entry point
├── Makefile                          # Install, test, lint, update, uninstall
│
├── src/ideanator/
│   ├── __init__.py                   # Package version
│   ├── exceptions.py                 # Custom exception hierarchy
│   ├── types.py                      # Pydantic BaseModel types + enums
│   ├── models.py                     # Pydantic models for refactoring pipeline
│   ├── config.py                     # Pydantic Settings + backend configs
│   ├── prompts.py                    # YAML loader with LRU cache
│   ├── scorer.py                     # Inverted vagueness assessment
│   ├── phases.py                     # Phase selection + prompt template builder
│   ├── parser.py                     # [REFLECTION]/[QUESTION] parsing + generic check
│   ├── pipeline.py                   # ARISE orchestration + refactoring engine integration
│   ├── refactor.py                   # Three-stage refactoring engine (extract/synthesize/validate)
│   ├── llm.py                        # LLMClient protocol + server managers
│   ├── cli.py                        # Click entry point + rich error handling
│   │
│   └── data/                         # Runtime data files (bundled in wheel)
│       ├── prompts.yaml              # ARISE questioning prompts
│       └── prompts/                  # Three-stage refactoring engine configs
│           ├── extract.yml           # Stage 1: structured extraction prompt
│           ├── synthesize.yml        # Stage 2: chain-of-density synthesis + banned words
│           └── validate.yml          # Stage 3: faithfulness/completeness/sycophancy checks
│
└── tests/
    ├── conftest.py                   # MockLLMClient + shared fixtures
    ├── test_backends.py              # Backend enum, config, server factory (16 tests)
    ├── test_cli.py                   # CLI flags, help, batch/interactive (30 tests)
    ├── test_config.py                # Pydantic settings, validation, env vars (25 tests)
    ├── test_exceptions.py            # Custom exception hierarchy (18 tests)
    ├── test_parser.py                # Response parsing + generic detection (29 tests)
    ├── test_phases.py                # Phase determination + prompt building (17 tests)
    ├── test_pipeline.py              # End-to-end pipeline integration (10 tests)
    ├── test_prompts.py               # YAML content integrity (23 tests)
    ├── test_refactor.py              # Refactoring engine: models, stages, status (29 tests)
    └── test_scorer.py                # Vagueness scoring + safety net (8 tests)
```

### Data flow

```
CLI (cli.py)
 │
 ├─ Resolve backend (--ollama / --mlx / --external)
 ├─ Get BackendConfig (model, URL, needs_server)
 ├─ Start server if needed (OllamaServer / MLXServer)
 ├─ Create OpenAILocalClient
 │
 └─► PIPELINE (pipeline.py)
      │
      ├─ 1. VAGUENESS SCORING (scorer.py)
      │     │  LLM call: "List what is MISSING" (temp=0.0, tokens=200)
      │     │  Parse dimension names from response
      │     │  Safety net: <20 words + "NONE" → all missing
      │     └─► DimensionCoverage (6 booleans)
      │
      ├─ 2. PHASE SELECTION (phases.py)
      │     │  Anchor: always
      │     │  Reveal: if core_problem or target_audience missing
      │     │  Imagine: if success_vision missing
      │     │  Scope: always
      │     └─► list[Phase]
      │
      ├─ 3. PHASE LOOP (for each phase)
      │     │
      │     ├─ a) Build prompt (phases.py)
      │     │     Template + {still_need} + random few-shot example
      │     │
      │     ├─ b) LLM call: interviewer (temp=0.6, tokens=250)
      │     │     user_message = raw idea (anchor) or conversation log (others)
      │     │
      │     ├─ c) Parse response (parser.py)
      │     │     Extract [REFLECTION], [QUESTION 1], [QUESTION 2]
      │     │     Graceful fallback to raw text if format not followed
      │     │
      │     ├─ d) Generic check (parser.py)
      │     │     Per-question keyword overlap with original idea
      │     │
      │     ├─ e) User response
      │     │     Batch: LLM-simulated (temp=0.7, tokens=200)
      │     │     Interactive: real user input via callback
      │     │
      │     └─ f) Update dimensions
      │           Mark phase's mapped dimensions as covered
      │
      ├─ 4. LEGACY SYNTHESIS (temp=0.3, tokens=500)
      │     Full conversation → structured summary with 8 headers
      │
      └─ 5. THREE-STAGE REFACTORING (refactor.py)
            │
            ├─ Stage 1: EXTRACT (temp=0.3, tokens=800)
            │  Conversation → ExtractedInsights (Pydantic model)
            │  Structured dimensions + user quotes + contradictions
            │
            ├─ Stage 2: SYNTHESIZE (temp=0.5, tokens=600)
            │  ExtractedInsights → refined statement (6 sections)
            │  Chain-of-density + banned words + first-person voice
            │
            ├─ Stage 3: VALIDATE (temp=0.2, tokens=600)
            │  Statement + transcript → ValidationResult
            │  Faithfulness + completeness + sycophancy checks
            │
            ├─ Self-refine loop (if confidence < 0.8, max 2 rounds)
            │  Critique → Stage 2 → Stage 3 → check again
            │
            ├─ Exploration status (programmatic, not LLM)
            │  Count user response words per dimension
            │
            ├─ Contradiction detection (programmatic + LLM)
            │  Negation patterns across phases
            │
            └─► RefactoredIdea
                  ├── one_liner, problem, solution, audience, differentiator
                  ├── open_questions
                  ├── exploration_status (per-dimension labels)
                  ├── contradictions_found
                  ├── validation (confidence, faithfulness, completeness, sycophancy)
                  └── refinement_rounds
```

### Module reference

#### `types.py` — Foundation types

Everything in the system builds on these types:

| Type | Purpose |
|------|---------|
| `Phase` | Enum: `anchor`, `reveal`, `imagine`, `scope` |
| `Dimension` | Enum: 6 vagueness dimensions (personal_motivation, target_audience, ...) |
| `DimensionCoverage` | Tracks True/False per dimension. Has `covered_count`, `uncovered_labels()`, `mark_covered()` |
| `ParsedResponse` | Structured LLM output: `reflection`, `question_1`, `question_2`, `raw`, `clean` |
| `GenericFlag` | Records flagged questions: `phase`, `question`, `flag` |
| `ConversationTurn` | Single turn: `phase`, `role`, `content`, `parsed` |
| `IdeaResult` | Complete pipeline output for one idea (includes `refactored` field) |

Key mapping — `PHASE_DIMENSION_MAP` defines which dimensions each phase unlocks:

```
Anchor  → personal_motivation, target_audience
Reveal  → core_problem
Imagine → success_vision
Scope   → constraints_risks, differentiation
```

After a phase runs, its mapped dimensions are marked as covered regardless of the model's actual response. The act of asking is what counts.

#### `models.py` — Pydantic models for the refactoring pipeline

Enforces structural correctness between pipeline stages:

| Model | Purpose |
|-------|---------|
| `ExtractedInsights` | Stage 1 output: problem, audience, solution, differentiation, motivation, key_phrases, contradictions, user_register, unresolved |
| `ValidationResult` | Stage 3 output: faithfulness, completeness, sycophancy checks + confidence score + critique |
| `ExplorationStatus` | Per-dimension labels: well_explored, partially_explored, not_explored |
| `Contradiction` | Detected inconsistency: earlier statement, later statement, turn references |
| `RefactoredIdea` | Complete refactoring output: 6 sections + all pipeline metadata |

#### `refactor.py` — Three-stage refactoring engine

| Function | What it does |
|----------|-------------|
| `extract(client, transcript)` | Stage 1: parse conversation into `ExtractedInsights` with turn citations |
| `synthesize(client, insights, transcript, critique?)` | Stage 2: chain-of-density synthesis with banned words |
| `validate(client, statement, transcript)` | Stage 3: faithfulness/completeness/sycophancy → `ValidationResult` |
| `refactor_idea(client, transcript, conversation, phases, callback?)` | Full pipeline with self-refine loop |
| `compute_exploration_status(conversation, phases)` | Programmatic per-dimension status from conversation structure |
| `detect_contradictions(conversation)` | Heuristic contradiction detection across user turns |
| `parse_synthesis_output(raw)` | Parse 6-section output into `RefactoredIdea` |
| `format_exploration_status(status)` | Format status with emoji labels for display |

#### `exceptions.py` — Exception hierarchy

All custom exceptions inherit from `IdeanatorError` for easy catching:

| Exception | Purpose |
|-----------|---------|
| `IdeanatorError` | Base exception — catches all ideanator errors |
| `ConfigurationError` | Configuration loading or validation failed |
| `ServerError` | LLM server start/stop/communication failed |
| `ValidationError` | Input validation failed |
| `PromptLoadError` | Failed to load prompt templates |
| `RefactoringError` | Refactoring pipeline failed |
| `ParseError` | Failed to parse LLM response |

Each exception carries a `message` and optional `details` dict for structured error context.

#### `config.py` — Configuration system

Configuration uses both frozen dataclasses (for immutable runtime config) and Pydantic Settings (for environment variable support with `IDEANATOR_*` prefix):

```python
@dataclass(frozen=True)
class TemperatureConfig:
    decision: float = 0.0      # Vagueness scoring — deterministic
    questioning: float = 0.6   # Phase questions — creative but grounded
    synthesis: float = 0.3     # Final summary — structured
    simulation: float = 0.7    # Simulated user — realistic variation

@dataclass(frozen=True)
class TokenConfig:
    decision: int = 200        # Short dimension list
    question: int = 250        # Two questions + reflection
    synthesis: int = 500       # Full structured summary
    simulation: int = 200      # Brief user-like response
```

The `BackendConfig` pairs each backend with its default model, URL, and whether it needs a managed server process.

#### `scorer.py` — Inverted vagueness assessment

The core innovation: asking a small LLM "what is MISSING?" instead of "what is present?"

```python
def assess_vagueness(client, idea, vagueness_prompt):
    # 1. Call LLM with inverted prompt
    raw = client.call(vagueness_prompt, idea, temperature=0.0, max_tokens=200)

    # 2. Parse: if dimension name appears in output, it's missing
    coverage = DimensionCoverage()
    for dim in Dimension:
        if dim.value in raw.lower():
            coverage.coverage[dim] = False    # Missing

    # 3. Safety net: short idea + "NONE" → override all to missing
    if "NONE" in raw and len(idea.split()) < 20:
        for dim in Dimension:
            coverage.coverage[dim] = False

    return coverage, raw
```

**Why inverted?** Small models (3B parameters) trained with RLHF tend to be sycophantic — they'll confirm that a 10-word idea "clearly covers all aspects." Asking what's *missing* exploits the opposite tendency: models love finding gaps and giving feedback.

#### `parser.py` — Response parsing

The LLM is instructed to output in `[REFLECTION]...[QUESTION 1]...[QUESTION 2]...` format. The parser uses three regex patterns to extract each section:

```python
re.search(r"\[REFLECTION\]\s*(.*?)(?=\[QUESTION|\Z)", text, re.DOTALL)
re.search(r"\[QUESTION 1\]\s*(.*?)(?=\[QUESTION 2\]|\Z)", text, re.DOTALL)
re.search(r"\[QUESTION 2\]\s*(.*)", text, re.DOTALL)
```

If the model doesn't follow the format (common with smaller models), the parser falls back gracefully — the raw text is used as-is, and the pipeline continues without interruption.

**Anti-generic heuristic:**

```python
def is_question_generic(question, idea):
    keywords = {w.strip(".,!?").lower() for w in idea.split()
                if len(w.strip(".,!?")) >= 4 and w.lower() not in STOP_WORDS}
    return not any(kw in question.lower() for kw in keywords)
```

The `STOP_WORDS` set (28 words) filters out common verbs and pronouns that would create false matches: "want", "build", "create", "people", "would", "their", etc.

#### `pipeline.py` — Orchestration

Two entry points:

| Function | Mode | User responses |
|----------|------|----------------|
| `run_arise_for_idea()` | Batch | LLM-simulated |
| `run_arise_interactive()` | Interactive | Real user via callback |

The callback system decouples progress reporting from pipeline logic:

```python
# Callback signature
Callable[[str, str], str | None]

# Events emitted
"status"       → Progress messages (including refactoring stage updates)
"vagueness"    → Vagueness assessment result
"phase_start"  → Phase label
"interviewer"  → Interviewer's questions
"user_sim"     → Simulated user response (batch only)
"generic_flag" → Flagged generic question
"prompt_user"  → Request for user input (interactive only)
"synthesis"    → Legacy synthesis text
"refactored"   → Three-stage refined idea statement
```

**Critical behavioral contracts:**
1. Anchor receives the raw idea as `user_message`; all other phases receive the full conversation log
2. Dimensions are updated after the phase runs (phase-based, not response-based)
3. Legacy synthesis `user_message` is the literal string `"Please synthesize now."`
4. A random few-shot example is selected per phase call (prevents deterministic output)
5. After legacy synthesis, the three-stage refactoring engine runs automatically
6. Exploration status is computed programmatically from conversation structure, not by LLM self-assessment

### Design patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Protocol (duck typing)** | `LLMClient` in `llm.py` | Enables mock-based testing without inheritance |
| **Factory** | `create_server()` in `llm.py` | Dispatches to backend-specific server manager |
| **Context Manager** | `MLXServer`, `OllamaServer` | Automatic server start/stop lifecycle |
| **Callback/Observer** | `pipeline.py` | Decouples CLI output from pipeline logic |
| **Frozen Dataclass** | `config.py` | Prevents accidental config mutation mid-run |
| **Pydantic Settings** | `config.py` | Environment variable support with `IDEANATOR_*` prefix |
| **Exception Hierarchy** | `exceptions.py` | Structured error handling with context details |
| **Pydantic Validation** | `models.py`, `types.py` | Structural correctness between pipeline stages |
| **LRU Cache** | `prompts.py`, `refactor.py` | Single YAML read, cached for entire session |
| **Template Method** | `phases.py` | Phase prompts follow consistent format pattern |
| **Graceful Degradation** | `parser.py`, `refactor.py` | Falls back to raw text if parsing fails |
| **Self-Refine Loop** | `refactor.py` | Critique-driven revision when quality is below threshold |
| **Strategy** | `config.py` | Different temperature/token configs per call type |

### Configuration system

Configuration is centralized in `config.py` using frozen dataclasses for immutable runtime config and Pydantic Settings for environment variable support (`IDEANATOR_*` prefix, `.env` file loading). This prevents a common bug in LLM pipelines: accidentally mutating temperature or token limits during multi-phase execution.

```python
# These are immutable — any attempt to modify raises AttributeError
TEMPERATURES = TemperatureConfig()
TOKENS = TokenConfig()
```

Backend configuration follows the same pattern:

```python
BACKEND_DEFAULTS = {
    Backend.OLLAMA:   BackendConfig("llama3.2:3b",             "http://localhost:11434/v1", needs_server=True),
    Backend.MLX:      BackendConfig("mlx-community/Llama-...", "http://localhost:8080/v1",  needs_server=True),
    Backend.EXTERNAL: BackendConfig("default",                 "http://localhost:8080/v1",  needs_server=False),
}
```

CLI flags (`-m`, `--server-url`) override these defaults at runtime.

### LLM abstraction layer

The `LLMClient` protocol is the central abstraction that makes the entire pipeline testable:

```python
@runtime_checkable
class LLMClient(Protocol):
    def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str: ...
```

Every module that talks to an LLM receives this protocol — never a concrete `OpenAI` instance. The real implementation (`OpenAILocalClient`) wraps the OpenAI Python client pointed at a local server. On failure, it raises `ServerError` with structured details (backend, model, original error) instead of returning raw error strings.

The OpenAI client is lazy-imported (only when `OpenAILocalClient` is instantiated), so importing `ideanator` doesn't require `openai` to be installed unless you actually use it.

### Server lifecycle management

Each backend that needs a managed server process implements the context manager pattern:

**OllamaServer:**
1. Checks if `ollama` binary exists (raises with install instructions if not)
2. Tests if the daemon is already running via HTTP health check (`/api/tags`)
3. Only starts the daemon if it's not already running (tracks via `_started_by_us` flag)
4. Pulls the requested model (`ollama pull` — no-op if already cached)
5. On exit, only terminates the daemon if we started it

**MLXServer:**
1. Spawns `python -m mlx_lm.server --model {model_id}` as a subprocess
2. Reads stdout line by line, waiting for `"Starting httpd"` (with timeout)
3. Sleeps 2 seconds for connection buffer
4. On exit, terminates the subprocess

Both follow the same interface, so the CLI dispatch logic is identical regardless of backend:

```python
if cfg.needs_server:
    with create_server(backend, model):
        client = OpenAILocalClient(base_url=url, model_id=model)
        _dispatch(client, ...)
else:
    client = OpenAILocalClient(base_url=url, model_id=model)
    _dispatch(client, ...)
```

### Pipeline internals

**LLM calls per idea (batch mode, all 4 phases):**

| # | Call | Temperature | Tokens | Purpose |
|---|------|-------------|--------|---------|
| 1 | Vagueness scoring | 0.0 | 200 | Assess what's missing |
| 2 | Anchor questions | 0.6 | 250 | Personal connection |
| 3 | Anchor simulation | 0.7 | 200 | Simulated user response |
| 4 | Reveal questions | 0.6 | 250 | Deeper problem |
| 5 | Reveal simulation | 0.7 | 200 | Simulated user response |
| 6 | Imagine questions | 0.6 | 250 | Success vision |
| 7 | Imagine simulation | 0.7 | 200 | Simulated user response |
| 8 | Scope questions | 0.6 | 250 | Reality check |
| 9 | Scope simulation | 0.7 | 200 | Simulated user response |
| 10 | Legacy synthesis | 0.3 | 500 | Structured summary |
| 11 | Extract | 0.3 | 800 | Parse into dimensions |
| 12 | Synthesize | 0.5 | 600 | Refined statement |
| 13 | Validate | 0.2 | 600 | Quality checks |

Total: **13 calls** per idea (batch, all phases). If validation triggers self-refine, add 2 calls per round (synthesize + validate).

**Conversation log format:**

Each interviewer turn is formatted as:
```
[Interviewer — Phase N — PHASE_NAME]:
{clean response text}
```

This format is important — subsequent phases receive the full conversation log as their `user_message`, and both the synthesis and refactoring engine use this format.

**Temperature strategy rationale:**

| Call type | Temperature | Reasoning |
|-----------|------------|-----------|
| Vagueness scoring | 0.0 | Deterministic — same idea should produce same assessment |
| Phase questions | 0.6 | Creative enough to ask good questions, grounded enough to stay relevant |
| Legacy synthesis | 0.3 | Structured summary — low creativity, high coherence |
| User simulation | 0.7 | Realistic variation — real users are unpredictable |
| Extraction | 0.3 | Deterministic parsing of conversation into structured data |
| Refined synthesis | 0.5 | Creative enough for natural language, grounded by extracted data |
| Validation | 0.2 | Near-deterministic quality checking |

**Output structure** (JSON):

```json
{
  "original_idea": "I want to build a language learning app.",
  "timestamp": "2025-01-15T10:30:00",
  "vagueness_assessment": {
    "dimensions": {"personal_motivation": false, "target_audience": false, ...},
    "score": "0/6",
    "uncovered": ["their personal motivation", ...],
    "raw_response": "PERSONAL_MOTIVATION\nTARGET_AUDIENCE\n..."
  },
  "phases_executed": ["anchor", "reveal", "imagine", "scope"],
  "conversation": [
    {"phase": "anchor", "role": "interviewer", "content": "..."},
    {"phase": "anchor", "role": "user_simulated", "content": "..."}
  ],
  "generic_flags": [
    {"phase": "scope", "question": "What challenges...", "flag": "GENERIC — could apply to any idea"}
  ],
  "synthesis": "[IDEA]: ...\n[WHO]: ...\n...",
  "refactored": {
    "one_liner": "I'm building a conversational Spanish practice tool for...",
    "problem": "College students trying to learn conversational Spanish...",
    "solution": "A practice tool that pairs learners with...",
    "audience": "College students taking Spanish courses...",
    "differentiator": "Unlike Duolingo and Babbel which sort by...",
    "open_questions": ["How would you measure improvement?", "..."],
    "exploration_status": {
      "problem": "well_explored",
      "audience": "well_explored",
      "solution": "partially_explored",
      "differentiation": "well_explored",
      "motivation": "well_explored"
    },
    "validation": {
      "confidence": 0.87,
      "faithfulness": {"supported_count": 6, "unsupported_count": 0, ...},
      "completeness": {"problem": true, "audience": true, ...},
      "sycophancy": {"severity": "none", "flags": []}
    },
    "contradictions": [],
    "refinement_rounds": 0,
    "extracted_insights": {
      "problem": "Current tools are too gamified [Turn 3]",
      "key_phrases": ["too gamified", "real conversations", ...],
      ...
    }
  }
}
```

### Testing strategy

The test suite (205 tests) runs entirely without a server. The `MockLLMClient` in `conftest.py` cycles through a list of predetermined responses:

```python
class MockLLMClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.calls = []

    def call(self, system_prompt, user_message, temperature, max_tokens):
        self.calls.append({...})
        result = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return result
```

This lets tests verify:
- **What was sent** to the LLM (system prompt content, temperature, token limits)
- **What the pipeline does** with each response (parsing, scoring, phase selection)
- **End-to-end behavior** (short idea -> 4 phases, detailed idea -> 2 phases)
- **Refactoring quality** (extraction parsing, synthesis output format, validation logic, self-refine loop)

Test coverage by module:

| Module | Tests | What's verified |
|--------|-------|----------------|
| `test_cli.py` | 30 | Help output, flag resolution, batch/interactive modes, backend defaults + overrides |
| `test_refactor.py` | 29 | Pydantic models, all three stages, exploration status, contradictions, output parsing, self-refine loop |
| `test_parser.py` | 29 | Structured parsing, fallback behavior, generic detection, stop words |
| `test_config.py` | 25 | Pydantic settings, backend config, environment variables, URL validation |
| `test_prompts.py` | 23 | YAML integrity — all placeholders, format markers, dimensions, example structure |
| `test_exceptions.py` | 18 | Custom exception hierarchy, message/details attributes, inheritance |
| `test_phases.py` | 17 | Phase determination, example selection, prompt building, uncovered truncation |
| `test_backends.py` | 16 | Backend enum, config defaults, server factory, init behavior |
| `test_pipeline.py` | 10 | End-to-end: phase counts, synthesis, refactored output, validation, exploration status, generic flags |
| `test_scorer.py` | 8 | All-missing, none-missing, safety net, partial dimensions, temperature/tokens |

The `_clear_caches` autouse fixture ensures the YAML cache, refactoring config cache, and settings singleton are reset between tests, preventing test pollution.

### Adding a new backend

To add a new backend (e.g., `llama.cpp`):

**1. Add to the enum** (`config.py`):
```python
class Backend(str, Enum):
    MLX = "mlx"
    OLLAMA = "ollama"
    EXTERNAL = "external"
    LLAMACPP = "llamacpp"        # new
```

**2. Add default config** (`config.py`):
```python
BACKEND_DEFAULTS[Backend.LLAMACPP] = BackendConfig(
    default_model="model.gguf",
    default_url="http://localhost:8080/v1",
    needs_server=True,
)
```

**3. Create server manager** (`llm.py`):
```python
class LlamaCppServer:
    def __init__(self, model_id, timeout=SERVER_STARTUP_TIMEOUT):
        self.model_id = model_id
        self.process = None
        # ...

    def start(self): ...
    def stop(self): ...
    def __enter__(self): self.start(); return self
    def __exit__(self, *a): self.stop()
```

**4. Register in factory** (`llm.py`):
```python
def create_server(backend, model_id):
    if backend == Backend.LLAMACPP:
        return LlamaCppServer(model_id=model_id)
    # ...
```

**5. Add CLI flag** (`cli.py`):
```python
@click.option("--llamacpp", "use_llamacpp", is_flag=True)
```

**6. Add tests** (`test_backends.py`).

No other modules need to change — the pipeline, scorer, parser, prompts, and refactoring engine are completely backend-agnostic.

---

## License

MIT
