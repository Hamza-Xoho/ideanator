"""Tests for the exception hierarchy."""

import pytest

from ideanator.exceptions import (
    ConfigurationError,
    IdeanatorError,
    ParseError,
    PromptLoadError,
    RefactoringError,
    ServerError,
    ValidationError,
)


class TestIdeanatorError:
    def test_message(self):
        err = IdeanatorError("something broke")
        assert str(err) == "something broke"
        assert err.message == "something broke"

    def test_details_default_empty(self):
        err = IdeanatorError("oops")
        assert err.details == {}

    def test_details_preserved(self):
        err = IdeanatorError("oops", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_is_exception(self):
        assert issubclass(IdeanatorError, Exception)


class TestSubclasses:
    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            ServerError,
            ValidationError,
            PromptLoadError,
            RefactoringError,
            ParseError,
        ],
    )
    def test_inherits_from_base(self, exc_class):
        assert issubclass(exc_class, IdeanatorError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            ServerError,
            ValidationError,
            PromptLoadError,
            RefactoringError,
            ParseError,
        ],
    )
    def test_can_be_caught_as_base(self, exc_class):
        with pytest.raises(IdeanatorError):
            raise exc_class("test error")

    def test_server_error_with_details(self):
        err = ServerError(
            "Cannot connect",
            details={"url": "http://localhost:8080", "model": "test"},
        )
        assert err.message == "Cannot connect"
        assert err.details["url"] == "http://localhost:8080"
        assert err.details["model"] == "test"

    def test_configuration_error(self):
        err = ConfigurationError("bad config")
        assert str(err) == "bad config"
