"""Chat-style message bubbles for the conversation view."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static


class MessageBubble(Widget):
    """A single message bubble — either from the interviewer or the user."""

    def __init__(
        self,
        body: str,
        *,
        sender: str = "",
        variant: str = "interviewer",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._body = body
        self._sender = sender
        self._variant = variant
        self.add_class(f"bubble-{variant}")

    def compose(self) -> ComposeResult:
        if self._sender:
            yield Static(self._sender, classes="bubble-sender")
        yield Static(self._body, classes="bubble-body", markup=True)


class PhaseHeader(Widget):
    """A thin phase-transition marker in the conversation stream."""

    def __init__(self, label: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold]{self._label}[/bold]",
            classes="phase-label-text",
            markup=True,
        )


class StatusLine(Widget):
    """A dim system status line (scoring, thinking, errors)."""

    def __init__(self, text: str, *, severity: str = "info", **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text
        self._severity = severity
        self.add_class(f"status-{severity}")

    def compose(self) -> ComposeResult:
        yield Static(self._text, classes="status-text", markup=True)
