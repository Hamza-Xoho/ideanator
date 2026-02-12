"""Horizontal phase progress indicator widget."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget

from rich.text import Text

_PHASE_COLORS: dict[str, str] = {
    "ANCHOR": "blue",
    "REVEAL": "magenta",
    "IMAGINE": "yellow",
    "SCOPE": "green",
}


class PhaseIndicator(Widget):
    """Shows phase progression: completed | current | upcoming."""

    current_index: reactive[int] = reactive(-1)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._phases: list[str] = []

    def set_phases(self, phase_names: list[str]) -> None:
        """Initialize with the list of phases that will run."""
        self._phases = phase_names
        self.refresh()

    def advance_to(self, index: int, label: str) -> None:
        """Move the indicator to the given phase index."""
        self.current_index = index

    def render(self) -> Text:
        if not self._phases:
            return Text.from_markup(
                "  [dim]Analyzing idea...[/dim]"
            )

        parts: list[str] = []
        for i, name in enumerate(self._phases):
            color = _PHASE_COLORS.get(name, "white")
            if i < self.current_index:
                # Completed
                parts.append(f"[{color} dim]{name}[/{color} dim]")
            elif i == self.current_index:
                # Current
                parts.append(f"[bold {color}]> {name}[/bold {color}]")
            else:
                # Upcoming
                parts.append(f"[dim]{name}[/dim]")

        progress = ""
        if self.current_index >= 0:
            progress = f"  ({self.current_index + 1}/{len(self._phases)})"

        return Text.from_markup("  " + "  ".join(parts) + progress)
