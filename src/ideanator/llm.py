"""LLM client abstraction and server lifecycle management for multiple backends."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from typing import Protocol, runtime_checkable

import httpx
from rich.console import Console

from ideanator.config import Backend, SERVER_STARTUP_TIMEOUT
from ideanator.exceptions import ServerError
from ideanator.parser import strip_thinking

logger = logging.getLogger(__name__)
console = Console()


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM interaction — enables testing with mocks."""

    def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str: ...


class OpenAILocalClient:
    """Wraps the OpenAI client pointing at any OpenAI-compatible server.

    Handles thinking-model output by:
    1. Checking backend-specific reasoning fields (reasoning, reasoning_content, thinking).
    2. Falling back to regex stripping of <think> blocks.
    3. Optionally disabling thinking via reasoning_effort=none for Ollama.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        timeout: float = 600.0,
        disable_thinking: bool = True,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key="local",
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=2,
        )
        self.model_id = model_id
        self.base_url = base_url
        self.disable_thinking = disable_thinking

    def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.6,
        max_tokens: int = 300,
    ) -> str:
        import openai

        extra_body: dict[str, str] = {}
        if self.disable_thinking and "ollama" in self.base_url.lower():
            extra_body["reasoning_effort"] = "none"

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
        except openai.APIConnectionError:
            logger.error("Cannot connect to LLM server at %s", self.base_url)
            raise ServerError(
                f"Cannot connect to LLM server at {self.base_url}. Is it running?",
                details={"url": self.base_url, "model": self.model_id},
            )
        except openai.APITimeoutError:
            logger.error("LLM request timed out")
            raise ServerError(
                "LLM request timed out. Model may be loading or overloaded.",
                details={"url": self.base_url, "model": self.model_id},
            )
        except openai.BadRequestError as e:
            logger.warning("Bad request: %s", e)
            raise ServerError(
                f"Bad request to LLM: {e}",
                details={"url": self.base_url, "model": self.model_id},
            ) from e
        except openai.APIError as e:
            logger.warning("API error: %s", e)
            raise ServerError(
                f"LLM API error: {e}",
                details={"url": self.base_url, "model": self.model_id},
            ) from e

        content = _extract_content(response)
        if not content:
            raise ServerError(
                "LLM returned empty response",
                details={"url": self.base_url, "model": self.model_id},
            )
        return content


def _extract_content(response: object) -> str:
    """Defensively extract content from an OpenAI-compatible response.

    Checks for backend-specific reasoning fields and strips thinking
    blocks as a safety net.
    """
    if not response.choices:  # type: ignore[union-attr]
        return ""

    msg = response.choices[0].message  # type: ignore[union-attr]
    if msg is None:
        return ""

    # Check if backend separated reasoning into a dedicated field
    reasoning = (
        getattr(msg, "reasoning", None)
        or getattr(msg, "reasoning_content", None)
        or getattr(msg, "thinking", None)
    )
    if reasoning:
        logger.debug("Thinking-model reasoning extracted (%d chars)", len(reasoning))

    content = msg.content or ""

    # Safety net: strip <think> blocks that leaked into content
    if "<think>" in content:
        content = strip_thinking(content)

    return content.strip()


# ── Server Protocol ────────────────────────────────────────────────────


class ServerManager(Protocol):
    """Protocol for server lifecycle management."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def __enter__(self) -> ServerManager: ...
    def __exit__(self, *args: object) -> None: ...


# ── MLX Backend (macOS only) ──────────────────────────────────────────


class MLXServer:
    """Context manager for MLX server lifecycle (macOS + Apple Silicon only)."""

    def __init__(self, model_id: str, timeout: int = SERVER_STARTUP_TIMEOUT) -> None:
        self.model_id = model_id
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Start the MLX LM server and wait for it to be ready."""
        logger.info("Starting MLX server with model: %s", self.model_id)

        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "mlx_lm.server", "--model", self.model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise ServerError(
                "MLX not found. Install: pip install mlx-lm"
            )

        start_time = time.time()
        for line in self.process.stdout:  # type: ignore[union-attr]
            if time.time() - start_time > self.timeout:
                self.stop()
                raise ServerError(
                    f"MLX server failed to start within {self.timeout}s"
                )
            if "Starting httpd" in line:
                logger.info("MLX server is ready")
                console.print("[green]\u2713[/green] MLX server started")
                break

        time.sleep(2)

    def stop(self) -> None:
        """Terminate the MLX server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("MLX server terminated")

    def __enter__(self) -> MLXServer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


# ── Ollama Backend (cross-platform) ───────────────────────────────────


class OllamaServer:
    """Context manager for Ollama server lifecycle (Linux, macOS, Windows)."""

    def __init__(self, model_id: str, timeout: int = SERVER_STARTUP_TIMEOUT) -> None:
        self.model_id = model_id
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._started_by_us = False

    def start(self) -> None:
        """Start Ollama if not already running, then pull the model."""
        if not shutil.which("ollama"):
            raise ServerError(
                "Ollama is not installed. Install from https://ollama.com\n"
                "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh\n"
                "  Windows: download from https://ollama.com/download"
            )

        if not self._is_running():
            logger.info("Starting Ollama daemon...")
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._started_by_us = True
            self._wait_for_ready()
            console.print("[green]\u2713[/green] Ollama server started")

        logger.info("Ensuring model is available: %s", self.model_id)
        pull_result = subprocess.run(
            ["ollama", "pull", self.model_id],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if pull_result.returncode != 0:
            raise ServerError(
                f"Failed to pull model '{self.model_id}': {pull_result.stderr}"
            )
        logger.info("Model ready: %s", self.model_id)

    def _is_running(self) -> bool:
        """Check if the Ollama daemon is responding."""
        try:
            import urllib.request

            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def _wait_for_ready(self) -> None:
        """Wait for the Ollama daemon to become responsive."""
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if self._is_running():
                logger.info("Ollama daemon is ready")
                return
            time.sleep(1)
        raise ServerError(
            f"Ollama daemon failed to start within {self.timeout}s"
        )

    def stop(self) -> None:
        """Terminate the Ollama daemon if we started it."""
        if self.process and self._started_by_us:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("Ollama daemon terminated")

    def __enter__(self) -> OllamaServer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


# ── Pre-flight Checks ─────────────────────────────────────────────────


def preflight_check(base_url: str, model_id: str, backend: Backend) -> bool:
    """Verify server connectivity and model availability before pipeline.

    Returns True if ready, False if there's a problem (with logged warnings).
    """
    if backend == Backend.EXTERNAL:
        return _check_server_health(base_url)

    if backend == Backend.OLLAMA:
        if not _check_server_health(base_url.replace("/v1", "")):
            return False
        return _check_ollama_model(model_id)

    # MLX — just check the server is responding
    return _check_server_health(base_url)


def _check_server_health(base_url: str) -> bool:
    """Quick connectivity check."""
    import urllib.request

    # Normalize: try /v1/models or just the base
    url = base_url.rstrip("/")
    try:
        req = urllib.request.Request(f"{url}/models")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        pass

    # Fallback: try the base URL itself
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        logger.warning("Cannot reach LLM server at %s", base_url)
        return False


def _check_ollama_model(model_id: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        import json
        import urllib.request

        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            available = {m["name"] for m in data.get("models", [])}
            if any(m == model_id or m.startswith(f"{model_id}:") for m in available):
                return True
            logger.warning(
                "Model '%s' not found in Ollama. Available: %s",
                model_id,
                ", ".join(sorted(available)[:5]),
            )
            return False
    except Exception as e:
        logger.warning("Could not check Ollama models: %s", e)
        return False


# ── Factory ───────────────────────────────────────────────────────────


def create_server(backend: Backend, model_id: str) -> ServerManager:
    """Create the appropriate server manager for the given backend."""
    if backend == Backend.MLX:
        return MLXServer(model_id=model_id)
    elif backend == Backend.OLLAMA:
        return OllamaServer(model_id=model_id)
    else:
        raise ValueError(
            f"Backend '{backend}' does not support auto-start. "
            f"Use --external with --server-url."
        )
