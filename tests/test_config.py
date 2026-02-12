"""Tests for configuration management."""

import pytest

from ideanator.config import (
    Backend,
    BackendConfig,
    BACKEND_DEFAULTS,
    Settings,
    TEMPERATURES,
    TOKENS,
    VAGUENESS_WORD_THRESHOLD,
    get_backend_config,
    get_settings,
    reload_settings,
)
from ideanator.exceptions import ConfigurationError


class TestTemperatureConfig:
    def test_decision_is_zero(self):
        assert TEMPERATURES.decision == 0.0

    def test_questioning_temperature(self):
        assert TEMPERATURES.questioning == 0.6

    def test_synthesis_temperature(self):
        assert TEMPERATURES.synthesis == 0.3

    def test_simulation_temperature(self):
        assert TEMPERATURES.simulation == 0.7

    def test_frozen(self):
        with pytest.raises(AttributeError):
            TEMPERATURES.decision = 0.5  # type: ignore[misc]


class TestTokenConfig:
    def test_decision_tokens(self):
        assert TOKENS.decision == 200

    def test_question_tokens(self):
        assert TOKENS.question == 250

    def test_synthesis_tokens(self):
        assert TOKENS.synthesis == 500

    def test_simulation_tokens(self):
        assert TOKENS.simulation == 200

    def test_frozen(self):
        with pytest.raises(AttributeError):
            TOKENS.decision = 100  # type: ignore[misc]


class TestVaguenessThreshold:
    def test_value(self):
        assert VAGUENESS_WORD_THRESHOLD == 20


class TestSettings:
    def test_default_ollama_url(self):
        s = Settings()
        assert s.ollama_url == "http://localhost:11434"

    def test_default_ollama_model(self):
        s = Settings()
        assert "qwen" in s.ollama_model.lower() or len(s.ollama_model) > 0

    def test_default_mlx_port(self):
        s = Settings()
        assert s.mlx_port == 8080

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_url_validation_rejects_invalid(self):
        with pytest.raises(Exception):
            Settings(ollama_url="not-a-url")

    def test_url_validation_accepts_http(self):
        s = Settings(ollama_url="http://myserver:1234")
        assert s.ollama_url == "http://myserver:1234"

    def test_url_validation_accepts_https(self):
        s = Settings(external_url="https://api.example.com")
        assert s.external_url == "https://api.example.com"

    def test_get_backend_url_ollama(self):
        s = Settings()
        assert "11434" in s.get_backend_url(Backend.OLLAMA)

    def test_get_backend_url_mlx(self):
        s = Settings()
        assert "8080" in s.get_backend_url(Backend.MLX)

    def test_get_backend_url_external(self):
        s = Settings(external_url="http://custom:9999")
        assert s.get_backend_url(Backend.EXTERNAL) == "http://custom:9999"

    def test_get_backend_model_ollama(self):
        s = Settings()
        assert len(s.get_backend_model(Backend.OLLAMA)) > 0

    def test_get_backend_model_mlx(self):
        s = Settings()
        assert len(s.get_backend_model(Backend.MLX)) > 0

    def test_get_settings_returns_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_reload_settings_returns_fresh(self):
        s1 = get_settings()
        s2 = reload_settings()
        assert isinstance(s2, Settings)
