"""Compact dimension coverage pills for the chat top bar."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget

from rich.text import Text

from ideanator.types import Dimension, Phase, PHASE_DIMENSION_MAP

_DIM_SHORT: dict[Dimension, str] = {
    Dimension.PERSONAL_MOTIVATION: "MOT",
    Dimension.TARGET_AUDIENCE: "AUD",
    Dimension.CORE_PROBLEM: "PRB",
    Dimension.SUCCESS_VISION: "VIS",
    Dimension.CONSTRAINTS_RISKS: "RSK",
    Dimension.DIFFERENTIATION: "DIF",
}

_PHASE_ORDER = [Phase.ANCHOR, Phase.REVEAL, Phase.IMAGINE, Phase.SCOPE]


class DimensionTracker(Widget):
    """Compact row of dimension pills: covered ones light up."""

    covered: reactive[frozenset] = reactive(frozenset)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._completed_phase_count: int = 0

    def mark_phase_complete(self, next_phase_index: int) -> None:
        """Mark dimensions covered by all phases completed so far."""
        new_covered: set[Dimension] = set(self.covered)
        for i in range(self._completed_phase_count, next_phase_index):
            if i < len(_PHASE_ORDER):
                phase = _PHASE_ORDER[i]
                for dim in PHASE_DIMENSION_MAP.get(phase, []):
                    new_covered.add(dim)
        self._completed_phase_count = next_phase_index
        self.covered = frozenset(new_covered)

    def mark_all_done(self) -> None:
        """Mark all dimensions covered (pipeline complete)."""
        self.covered = frozenset(Dimension)

    def render(self) -> Text:
        pills: list[str] = []
        for dim in Dimension:
            tag = _DIM_SHORT[dim]
            if dim in self.covered:
                pills.append(f"[black on green] {tag} [/black on green]")
            else:
                pills.append(f"[white on #333333] {tag} [/white on #333333]")
        count = len(self.covered)
        return Text.from_markup(" ".join(pills) + f"  [dim]{count}/6[/dim]")
