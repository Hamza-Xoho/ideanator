"""Tests for backend configuration and server factory."""

import pytest

from ideanator.config import Backend, BackendConfig, get_backend_config, BACKEND_DEFAULTS
from ideanator.llm import MLXServer, OllamaServer, create_server


class TestBackendEnum:
    def test_mlx_value(self):
        assert Backend.MLX.value == "mlx"

    def test_ollama_value(self):
        assert Backend.OLLAMA.value == "ollama"

    def test_external_value(self):
        assert Backend.EXTERNAL.value == "external"

    def test_from_string(self):
        assert Backend("mlx") == Backend.MLX
        assert Backend("ollama") == Backend.OLLAMA
        assert Backend("external") == Backend.EXTERNAL


class TestBackendConfig:
    def test_mlx_defaults(self):
        cfg = get_backend_config(Backend.MLX)
        assert "mlx-community" in cfg.default_model
        assert "8080" in cfg.default_url
        assert cfg.needs_server is True

    def test_ollama_defaults(self):
        cfg = get_backend_config(Backend.OLLAMA)
        assert "llama3.2" in cfg.default_model
        assert "11434" in cfg.default_url
        assert cfg.needs_server is True

    def test_external_defaults(self):
        cfg = get_backend_config(Backend.EXTERNAL)
        assert cfg.needs_server is False

    def test_all_backends_have_configs(self):
        for backend in Backend:
            cfg = get_backend_config(backend)
            assert isinstance(cfg, BackendConfig)
            assert len(cfg.default_model) > 0
            assert len(cfg.default_url) > 0

    def test_config_is_frozen(self):
        cfg = get_backend_config(Backend.MLX)
        with pytest.raises(AttributeError):
            cfg.default_model = "something-else"


class TestServerFactory:
    def test_create_mlx_server(self):
        server = create_server(Backend.MLX, "test-model")
        assert isinstance(server, MLXServer)
        assert server.model_id == "test-model"

    def test_create_ollama_server(self):
        server = create_server(Backend.OLLAMA, "test-model")
        assert isinstance(server, OllamaServer)
        assert server.model_id == "test-model"

    def test_create_external_raises(self):
        with pytest.raises(ValueError, match="does not support auto-start"):
            create_server(Backend.EXTERNAL, "test-model")


class TestOllamaServer:
    def test_init(self):
        server = OllamaServer(model_id="llama3.2:3b")
        assert server.model_id == "llama3.2:3b"
        assert server.process is None
        assert server._started_by_us is False

    def test_stop_only_terminates_if_started_by_us(self):
        """If Ollama was already running, stop() should not terminate it."""
        server = OllamaServer(model_id="test")
        server._started_by_us = False
        server.stop()  # Should not raise
        assert server.process is None


class TestMLXServer:
    def test_init(self):
        server = MLXServer(model_id="mlx-community/test-model")
        assert server.model_id == "mlx-community/test-model"
        assert server.process is None

    def test_stop_without_start(self):
        """Stopping a server that was never started should not raise."""
        server = MLXServer(model_id="test")
        server.stop()  # Should not raise
