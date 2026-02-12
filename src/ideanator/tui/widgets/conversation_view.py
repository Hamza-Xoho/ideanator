"""Scrollable conversation stream built from MessageBubble widgets."""

from __future__ import annotations

from textual.containers import VerticalScroll

from ideanator.tui.widgets.message_bubble import (
    MessageBubble,
    PhaseHeader,
    StatusLine,
)


class ConversationView(VerticalScroll):
    """Scrollable chat feed — mounts bubble widgets as messages arrive."""

    def add_status(self, text: str, *, severity: str = "info") -> None:
        """Append a dim status line."""
        self.mount(StatusLine(text, severity=severity))
        self.scroll_end(animate=False)

    def add_phase_header(self, label: str) -> None:
        """Append a phase transition marker."""
        self.mount(PhaseHeader(label))
        self.scroll_end(animate=False)

    def add_interviewer_message(self, text: str) -> None:
        """Append an interviewer message bubble."""
        self.mount(
            MessageBubble(text, sender="Interviewer", variant="interviewer")
        )
        self.scroll_end(animate=True)

    def add_user_message(self, text: str) -> None:
        """Append a user message bubble (right-aligned)."""
        self.mount(
            MessageBubble(text, sender="You", variant="user")
        )
        self.scroll_end(animate=True)

    def add_simulated_response(self, text: str) -> None:
        """Append a simulated user response bubble (batch mode)."""
        self.mount(
            MessageBubble(text, sender="Simulated User", variant="simulated")
        )
        self.scroll_end(animate=True)

    def add_warning(self, text: str) -> None:
        """Append a yellow warning line."""
        self.mount(StatusLine(text, severity="warning"))
        self.scroll_end(animate=False)

    def add_error(self, text: str) -> None:
        """Append a red error line."""
        self.mount(StatusLine(text, severity="error"))
        self.scroll_end(animate=False)
