"""LLM client abstraction and MLX server lifecycle management."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Protocol, runtime_checkable

from ideanator.config import SERVER_STARTUP_TIMEOUT

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
    """Wraps the OpenAI client pointing at a local MLX server."""

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


class MLXServer:
    """Context manager for MLX server lifecycle."""

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
