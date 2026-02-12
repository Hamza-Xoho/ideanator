"""Tests for PipelineWorker and BatchPipelineWorker.

Focuses on thread synchronization, callback routing, and error handling.
Skipped entirely if textual is not installed.
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from ideanator.tui.worker import PipelineWorker, BatchPipelineWorker
    from ideanator.tui.messages import (
        PipelineError,
        PipelineStatus,
        UserPromptRequested,
    )

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

pytestmark = pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")


class TestPipelineWorkerSync:
    """PipelineWorker user-input synchronization."""

    def test_submit_user_response(self):
        target = MagicMock()
        worker = PipelineWorker(target)
        worker.submit_user_response("hello")
        assert worker._user_response == "hello"
        assert worker._input_ready.is_set()

    def test_cancel_sets_flag_and_unblocks(self):
        target = MagicMock()
        worker = PipelineWorker(target)
        worker.cancel()
        assert worker._cancelled is True
        assert worker._input_ready.is_set()

    def test_callback_status_posts_message(self):
        target = MagicMock()
        worker = PipelineWorker(target)
        worker._callback("status", "Loading model...")
        target.post_message.assert_called_once()
        msg = target.post_message.call_args[0][0]
        assert isinstance(msg, PipelineStatus)
        assert msg.text == "Loading model..."

    def test_callback_prompt_user_blocks_then_returns(self):
        """prompt_user should block until submit_user_response is called."""
        target = MagicMock()
        worker = PipelineWorker(target)

        result_holder = []

        def call_callback():
            r = worker._callback("prompt_user", "ANCHOR")
            result_holder.append(r)

        t = threading.Thread(target=call_callback)
        t.start()

        # Give the thread time to start and block
        import time
        time.sleep(0.1)

        worker.submit_user_response("my answer")
        t.join(timeout=2)

        assert result_holder == ["my answer"]
        # Should have posted a UserPromptRequested
        posted_msg = target.post_message.call_args[0][0]
        assert isinstance(posted_msg, UserPromptRequested)

    def test_callback_cancelled_returns_none(self):
        target = MagicMock()
        worker = PipelineWorker(target)
        worker.cancel()
        result = worker._callback("status", "anything")
        assert result is None
        target.post_message.assert_not_called()


class TestBatchPipelineWorkerErrors:
    """BatchPipelineWorker file validation and error handling."""

    def test_missing_file(self):
        target = MagicMock()
        worker = BatchPipelineWorker(target)
        worker.run(
            file_path="/nonexistent/path/ideas.json",
            output_path="out.json",
            backend=MagicMock(),
            model="test",
            server_url="http://localhost:8080/v1",
        )
        target.post_message.assert_called_once()
        msg = target.post_message.call_args[0][0]
        assert isinstance(msg, PipelineError)
        assert "not found" in msg.error.lower() or "File not found" in msg.error

    def test_invalid_json(self):
        target = MagicMock()
        worker = BatchPipelineWorker(target)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json{{{")
            f.flush()
            worker.run(
                file_path=f.name,
                output_path="out.json",
                backend=MagicMock(),
                model="test",
                server_url="http://localhost:8080/v1",
            )
        msg = target.post_message.call_args[0][0]
        assert isinstance(msg, PipelineError)
        assert "invalid json" in msg.error.lower() or "JSON" in msg.error

    def test_missing_ideas_key(self):
        target = MagicMock()
        worker = BatchPipelineWorker(target)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"items": []}, f)
            f.flush()
            worker.run(
                file_path=f.name,
                output_path="out.json",
                backend=MagicMock(),
                model="test",
                server_url="http://localhost:8080/v1",
            )
        msg = target.post_message.call_args[0][0]
        assert isinstance(msg, PipelineError)
        assert "ideas" in msg.error.lower()

    def test_empty_ideas_list(self):
        target = MagicMock()
        worker = BatchPipelineWorker(target)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"ideas": []}, f)
            f.flush()
            worker.run(
                file_path=f.name,
                output_path="out.json",
                backend=MagicMock(),
                model="test",
                server_url="http://localhost:8080/v1",
            )
        msg = target.post_message.call_args[0][0]
        assert isinstance(msg, PipelineError)
        assert "no ideas" in msg.error.lower()

    def test_cancel(self):
        target = MagicMock()
        worker = BatchPipelineWorker(target)
        worker.cancel()
        assert worker._cancelled is True
