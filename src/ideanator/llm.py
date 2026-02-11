"""LLM client abstraction and server lifecycle management for multiple backends."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from typing import Protocol, runtime_checkable

from ideanator.config import Backend, SERVER_STARTUP_TIMEOUT

logger = logging.getLogger(__name__)


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
    """Wraps the OpenAI client pointing at any OpenAI-compatible server."""

    def __init__(self, base_url: str, model_id: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="local")
        self.model_id = model_id

    def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.6,
        max_tokens: int = 300,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Model call failed: %s", e)
            return f"[ERROR: {e}]"


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

    def __init__(self, model_id: str, timeout: int = SERVER_STARTUP_TIMEOUT):
        self.model_id = model_id
        self.timeout = timeout
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        """Start the MLX LM server and wait for it to be ready."""
        logger.info("Starting MLX server with model: %s", self.model_id)

        self.process = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm.server", "--model", self.model_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        start_time = time.time()
        for line in self.process.stdout:
            if time.time() - start_time > self.timeout:
                self.stop()
                raise TimeoutError(
                    f"MLX server failed to start within {self.timeout}s"
                )
            if "Starting httpd" in line:
                logger.info("MLX server is ready")
                break

        time.sleep(2)

    def stop(self) -> None:
        """Terminate the MLX server process."""
        if self.process:
            self.process.terminate()
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

    def __init__(self, model_id: str, timeout: int = SERVER_STARTUP_TIMEOUT):
        self.model_id = model_id
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._started_by_us = False

    def start(self) -> None:
        """
        Start Ollama if not already running, then pull the model.

        Ollama's architecture: `ollama serve` runs a background daemon,
        `ollama pull` downloads models, and the daemon exposes an
        OpenAI-compatible API at http://localhost:11434/v1.
        """
        if not shutil.which("ollama"):
            raise RuntimeError(
                "Ollama is not installed. Install from https://ollama.com\n"
                "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh\n"
                "  Windows: download from https://ollama.com/download"
            )

        # Check if Ollama is already running by testing the API
        if not self._is_running():
            logger.info("Starting Ollama daemon...")
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._started_by_us = True
            self._wait_for_ready()

        # Pull the model (no-op if already downloaded)
        logger.info("Ensuring model is available: %s", self.model_id)
        pull_result = subprocess.run(
            ["ollama", "pull", self.model_id],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if pull_result.returncode != 0:
            raise RuntimeError(
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
        raise TimeoutError(
            f"Ollama daemon failed to start within {self.timeout}s"
        )

    def stop(self) -> None:
        """Terminate the Ollama daemon if we started it."""
        if self.process and self._started_by_us:
            self.process.terminate()
            self.process = None
            logger.info("Ollama daemon terminated")

    def __enter__(self) -> OllamaServer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


# ── Factory ───────────────────────────────────────────────────────────


def create_server(backend: Backend, model_id: str) -> ServerManager:
    """Create the appropriate server manager for the given backend."""
    if backend == Backend.MLX:
        return MLXServer(model_id=model_id)
    elif backend == Backend.OLLAMA:
        return OllamaServer(model_id=model_id)
    else:
        raise ValueError(f"Backend '{backend}' does not support auto-start. "
                         f"Use --backend external with --server-url.")
