# Fixing thinking models and hardening local LLM pipelines

**The core bug has a straightforward fix: strip `<think>` blocks before parsing, or better yet, tell the backend to separate them.** Ollama, vLLM, and LM Studio all now provide dedicated fields for reasoning content — but each uses a different field name, and the basic MLX server passes thinking tokens through raw. The deeper issue is that regex-based parsing of `[REFLECTION]`/`[QUESTION]` tags is inherently fragile with small local models, and switching to grammar-constrained JSON output via Ollama's `format` parameter eliminates an entire class of failures. This report covers the immediate thinking-model fix, a migration path to robust structured output, and defensive patterns for multi-backend CLI tools.

---

## How thinking models break your parser — and the three-line fix

Every major open-weight thinking model (DeepSeek R1, Qwen3/QwQ, Phi-4-reasoning, Cogito) uses the same `<think>...</think>` delimiter convention. When a user runs ideanator with one of these models, the raw `content` field contains both the chain-of-thought reasoning and the actual answer. The regex looking for `[REFLECTION]` may match text inside the thinking block, or the parser may find no tags at all because the model placed its structured output after hundreds of lines of reasoning.

**The immediate fix** is a three-line strip at the client level, before any parsing occurs:

```python
import re

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks, including unclosed tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)  # unclosed tag
    return re.sub(r'\n{3,}', '\n\n', text).strip()
```

The `re.DOTALL` flag is critical — thinking blocks are always multiline. The second regex handles the edge case where a model hits its token limit mid-thought and never emits `</think>`. This function should be called **immediately after receiving the API response**, before any structured output parsing, because stripping at the client level means all downstream code sees clean content. DeepSeek's own documentation warns that keeping `<think>` blocks in conversation history "hurts model performance and eats up context length very quickly."

However, regex stripping is a fallback. The better approach is to use the backend's native thinking separation.

## Backend-native separation is cleaner than regex

Modern backends separate thinking content into a dedicated response field, but **each backend uses a different field name** — a critical compatibility detail:

| Backend | Request parameter | Thinking field | Answer field |
|---------|------------------|---------------|-------------|
| Ollama `/v1/chat/completions` | `reasoning_effort` via `extra_body` | `message.reasoning` | `message.content` |
| Ollama native `/api/chat` | `think: true/false` | `message.thinking` | `message.content` |
| vLLM | `--enable-reasoning` server flag | `message.reasoning_content` | `message.content` |
| DeepSeek official API | `thinking.type: "enabled"` | `message.reasoning_content` | `message.content` |
| LM Studio (v0.3.9+) | App setting | `message.reasoning_content` | `message.content` |
| Basic `mlx_lm.server` | None | **Mixed into `content` with raw tags** | Same field |

The OpenAI Python client does not officially define `reasoning` or `reasoning_content` attributes, but it passes through extra fields from the server response. Access them defensively with `getattr`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="deepseek-r1",
    messages=[{"role": "user", "content": "Analyze this topic..."}],
    extra_body={"reasoning_effort": "none"}  # Disable thinking entirely
)

msg = response.choices[0].message
# Universal extraction: check all possible field names, then fall back to regex
reasoning = (
    getattr(msg, 'reasoning', None) or
    getattr(msg, 'reasoning_content', None) or
    getattr(msg, 'thinking', None)
)
content = msg.content
if content and '<think>' in content:
    content = strip_thinking(content)  # Fallback for raw-pass-through backends
```

To **disable thinking entirely** through Ollama's OpenAI-compatible endpoint, pass `"reasoning_effort": "none"` in `extra_body`. This tells the model not to reason at all, which is likely what you want for a structured-output pipeline where the thinking just wastes tokens and creates parsing risk. Qwen3 has a known quirk: even with thinking disabled, it may emit empty `<think>\n\n</think>` blocks, so the regex strip remains necessary as a safety net.

## Replace regex tag parsing with grammar-constrained JSON

The thinking-model bug is actually a symptom of a deeper fragility: **regex-based parsing of custom tags is unreliable with small local models**. Models in the 1B–7B range fail to follow format instructions in characteristic ways — wrong tag casing (`[Reflection]` vs `[REFLECTION]`), missing tags entirely, echoing the format template verbatim, or using alternative delimiters like `**REFLECTION**`. IFEval benchmark data shows that even strong small models like Llama 3.2 3B only achieve **77.4%** instruction-following accuracy, and Gemma 2 2B drops to **61.9%**.

Ollama (v0.5+) supports **grammar-constrained decoding** that guarantees structurally valid output by masking invalid tokens during generation. This eliminates all structural failure modes:

```python
from ollama import chat
from pydantic import BaseModel, Field
from typing import List

class ARISEPhaseOutput(BaseModel):
    reflection: str = Field(description="A thoughtful reflection on the topic")
    questions: List[str] = Field(description="2-3 follow-up questions")

response = chat(
    model='qwen2.5:7b',
    messages=[{
        'role': 'system',
        'content': 'Analyze the given topic. Return JSON with a "reflection" string '
                   'and a "questions" array of 2-3 strings.'
    }, {
        'role': 'user',
        'content': f'Analyze: {topic}'
    }],
    format=ARISEPhaseOutput.model_json_schema(),
    options={'temperature': 0},
)
result = ARISEPhaseOutput.model_validate_json(response.message.content)
```

A critical detail: **the model does not see the `format` parameter**. Ollama uses it only for grammar masking at the token-sampling level. You must also describe the desired JSON structure in your prompt text, or the model will produce structurally valid but semantically empty JSON.

For the OpenAI-compatible endpoint, use `response_format` with the JSON schema:

```python
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "arise_output",
            "schema": ARISEPhaseOutput.model_json_schema()
        }
    }
)
```

If you must retain the tag-based format (e.g., for backward compatibility), implement **tiered fallback parsing** — strict regex first, then case-insensitive fuzzy regex, then heuristic content extraction:

```python
def parse_tags_fuzzy(text: str):
    """Case-insensitive, flexible delimiter matching."""
    reflection = re.search(
        r'(?:\[|<|\*\*)\s*[Rr]eflection\s*(?:\]|>|\*\*)\s*:?\s*(.*?)'
        r'(?=(?:\[|<|\*\*)\s*[Qq]uestion|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    questions = re.findall(
        r'(?:\[|<|\*\*)\s*[Qq]uestion\s*\d*\s*(?:\]|>|\*\*)\s*:?\s*(.*?)'
        r'(?=(?:\[|<|\*\*)\s*[Qq]uestion|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    return reflection, questions
```

The **`instructor` library** is the most production-ready wrapper for this pattern with Ollama. It handles automatic retries with Pydantic validation error feedback:

```python
import instructor

client = instructor.from_provider("ollama/qwen2.5:7b")
result = client.create(
    messages=[{"role": "user", "content": "Analyze: climate change impacts"}],
    response_model=ARISEPhaseOutput,
    max_retries=3,  # Auto-retries feeding validation errors back to the model
)
```

**Qwen 2.5 7B is the recommended model** for structured output at the small-model scale — it has the best balance of instruction following, tool calling support, and structured output compliance. Below 3B parameters, all models require grammar-constrained decoding for reliable structured output.

## Defensive patterns across Ollama, vLLM, llama.cpp, and LM Studio

When the OpenAI Python client talks to different backends, the response JSON varies in subtle but breaking ways. The `id` field uses `cmpl-` prefix in vLLM vs `chatcmpl-` everywhere else. The `system_fingerprint` is a real string from OpenAI but `null` or absent locally. LM Studio adds extra `stats`, `model_info`, and `runtime` fields. The MLX server defaults to only **500 max completion tokens** — surprisingly low and a common source of truncated output.

A defensive response normalizer handles all these differences:

```python
def safe_extract(response, backend_name: str) -> dict:
    """Normalize responses across any OpenAI-compatible backend."""
    choice = response.choices[0] if response.choices else None
    msg = choice.message if choice else None
    return {
        "content": strip_thinking(msg.content) if msg and msg.content else None,
        "reasoning": (
            getattr(msg, 'reasoning', None) or
            getattr(msg, 'reasoning_content', None)
        ) if msg else None,
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        "model": getattr(response, "model", "unknown"),
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0)
                         if response.usage else 0,
        "completion_tokens": getattr(response.usage, "completion_tokens", 0)
                             if response.usage else 0,
    }
```

Error formats also diverge. llama.cpp returns rich context-exceeded errors with `n_prompt_tokens` and `n_ctx` fields. vLLM may return an empty completion instead of an error when the prompt is too long. Ollama may silently truncate. The OpenAI Python client wraps all of these into typed exceptions (`openai.BadRequestError`, `openai.APIConnectionError`, etc.), so catch those rather than parsing raw HTTP responses:

```python
import openai

try:
    response = client.chat.completions.create(model=model, messages=messages)
except openai.BadRequestError as e:
    # Context exceeded, invalid params, etc.
    error_body = getattr(e, 'body', {})
    if isinstance(error_body, dict):
        err = error_body.get('error', {})
        if err.get('type') == 'exceed_context_size_error':
            # llama.cpp-specific: has n_prompt_tokens and n_ctx
            handle_context_overflow(err)
except openai.APIConnectionError:
    handle_server_down()
except openai.APITimeoutError:
    handle_slow_inference()
```

## Pre-flight checks and resource-aware pipeline execution

A multi-phase pipeline should verify everything before starting. Ollama exposes `GET /api/tags` for listing models and `POST /api/show` for model details:

```python
import httpx, sys

def preflight(base_url: str, required_models: list[str]) -> bool:
    """Verify server and models before pipeline starts."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        available = {m["name"] for m in resp.json().get("models", [])}
    except (httpx.ConnectError, httpx.TimeoutException):
        print("ERROR: Ollama not running. Start with: ollama serve")
        return False

    for model in required_models:
        if not any(m == model or m.startswith(f"{model}:") for m in available):
            print(f"ERROR: Model '{model}' not found. Run: ollama pull {model}")
            return False
    return True

if not preflight("http://localhost:11434", ["qwen2.5:7b"]):
    sys.exit(1)
```

For context window management — critical with 2K–4K token models — **token counting differs by ~20% across model families**. Use HuggingFace's `AutoTokenizer` for exact counts or `tiktoken` with `cl100k_base` encoding as a reasonable approximation for Llama 3 family models. Ollama's default context is **4096 tokens**, configurable via `OLLAMA_CONTEXT_LENGTH` environment variable or `num_ctx` in the Modelfile. A practical pattern for multi-phase pipelines is to summarize prior phase outputs before injecting them as context:

```python
def compress_prior_context(client, model, phase_outputs: list[str],
                           budget_tokens: int = 500) -> str:
    """Summarize previous phase results to fit context window."""
    combined = "\n---\n".join(phase_outputs)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": f"Summarize in under {budget_tokens} tokens, "
                              f"preserving key facts:\n\n{combined}"}],
        max_tokens=budget_tokens,
    )
    return resp.choices[0].message.content
```

Configure the OpenAI client for local inference with generous timeouts — large quantized models on CPU can take several minutes per response:

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=httpx.Timeout(600.0, connect=10.0, read=600.0, write=30.0),
    max_retries=3,
)
```

Ollama queues requests exceeding `OLLAMA_NUM_PARALLEL` (default: 4 or 1 depending on memory) in a FIFO queue up to 512 deep. For sequential pipeline phases this is fine, but if running parallel phases, bound concurrency with `asyncio.Semaphore`. Between phases, unload models via `keep_alive: 0` to free memory for the next model — this matters on memory-constrained systems where different pipeline phases use different models.

## Conclusion

The thinking-model bug has two clean solutions: either pass `"reasoning_effort": "none"` to disable thinking entirely (simplest for a structured-output pipeline), or extract the separated reasoning from the backend-specific field (`reasoning` for Ollama, `reasoning_content` for vLLM/LM Studio) and always apply regex stripping as a safety net. The larger opportunity is migrating from regex tag parsing to Pydantic schemas with Ollama's grammar-constrained `format` parameter, which eliminates structural parsing failures entirely and works down to 1.5B parameter models. The `instructor` library provides the cleanest abstraction for this with automatic retry and validation. For multi-backend resilience, the key patterns are defensive `getattr()` access on response fields, typed exception handling for divergent error formats, explicit `max_tokens` on every request (MLX's 500-token default will bite you otherwise), and pre-flight model availability checks before committing to a multi-phase pipeline run.