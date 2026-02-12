"""Settings screen — mirrors every CLI flag in a scrollable form."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Static,
)

from ideanator.config import (
    BACKEND_DEFAULTS,
    DEFAULT_OUTPUT_FILE,
    Backend,
)


@dataclass
class AppSettings:
    """Mutable settings object shared across the app.

    Maps 1-to-1 with the CLI flags:
        --ollama / --mlx / --external   →  backend
        -m, --model ID                  →  model
        --server-url URL                →  server_url
        -f, --file PATH                 →  batch_file
        -o, --output PATH               →  output_file
        -v, --verbose                   →  verbose
    """

    backend: Backend = Backend.OLLAMA
    model: str = ""
    server_url: str = ""
    batch_file: str = ""
    output_file: str = DEFAULT_OUTPUT_FILE
    verbose: bool = False


class SettingsScreen(Screen):
    """Settings form — every CLI flag as a TUI widget."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, settings: AppSettings, **kwargs) -> None:
        super().__init__(**kwargs)
        self._settings = settings

    # ── Layout ──────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="settings-scroll"):
            with Vertical(id="settings-panel"):
                yield Static("Settings", id="settings-title")

                # ── BACKENDS (pick one) ──
                yield Static("Backend", classes="section-label")
                yield Static(
                    "Pick one. Default: Ollama.",
                    classes="section-hint",
                )
                with RadioSet(id="backend-radio"):
                    yield RadioButton(
                        "Ollama  (Linux, macOS, Windows)",
                        value=self._settings.backend == Backend.OLLAMA,
                        id="rb-ollama",
                    )
                    yield RadioButton(
                        "MLX  (macOS + Apple Silicon)",
                        value=self._settings.backend == Backend.MLX,
                        id="rb-mlx",
                    )
                    yield RadioButton(
                        "External  (any already-running server)",
                        value=self._settings.backend == Backend.EXTERNAL,
                        id="rb-external",
                    )

                # ── OPTIONS ──
                yield Static("Options", classes="section-label section-divider")

                # -m, --model
                with Horizontal(classes="field-row"):
                    yield Static("-m, --model", classes="field-label")
                    yield Input(
                        value=self._settings.model,
                        placeholder=BACKEND_DEFAULTS[
                            self._settings.backend
                        ].default_model,
                        id="model-input",
                    )

                # --server-url
                with Horizontal(classes="field-row"):
                    yield Static("--server-url", classes="field-label")
                    yield Input(
                        value=self._settings.server_url,
                        placeholder=BACKEND_DEFAULTS[
                            self._settings.backend
                        ].default_url,
                        id="url-input",
                    )

                # -o, --output
                with Horizontal(classes="field-row"):
                    yield Static("-o, --output", classes="field-label")
                    yield Input(
                        value=self._settings.output_file,
                        placeholder=DEFAULT_OUTPUT_FILE,
                        id="output-input",
                    )

                # -v, --verbose
                yield Checkbox(
                    "Verbose logging  (-v)",
                    value=self._settings.verbose,
                    id="verbose-check",
                )

                # ── BATCH MODE ──
                yield Static(
                    "Batch Mode",
                    classes="section-label section-divider",
                )
                yield Static(
                    "Process multiple ideas from a JSON file with "
                    "simulated user responses. Useful for testing "
                    "prompt efficacy at scale.",
                    classes="section-hint",
                )

                # -f, --file
                with Horizontal(classes="field-row"):
                    yield Static("-f, --file", classes="field-label")
                    yield Input(
                        value=self._settings.batch_file,
                        placeholder='Path to JSON  (e.g. ideas.json)',
                        id="batch-file-input",
                    )

                yield Static(
                    'Format: { "ideas": [{"content": "..."}, ...] }',
                    classes="section-hint",
                )

                # ── ACTIONS ──
                with Horizontal(id="settings-actions"):
                    yield Button(
                        "Save", variant="primary", id="save-settings-btn"
                    )
                    yield Button(
                        "Reset Defaults",
                        variant="warning",
                        id="reset-settings-btn",
                    )
                    yield Button(
                        "Cancel", variant="default", id="cancel-settings-btn"
                    )

        yield Footer()

    # ── Events ──────────────────────────────────────────

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Update model/URL placeholders when backend changes."""
        backend = self._index_to_backend(event.radio_set.pressed_index)
        cfg = BACKEND_DEFAULTS[backend]
        self.query_one("#model-input", Input).placeholder = cfg.default_model
        self.query_one("#url-input", Input).placeholder = cfg.default_url

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-settings-btn":
            self._save_and_close()
        elif event.button.id == "reset-settings-btn":
            self._reset_defaults()
        elif event.button.id == "cancel-settings-btn":
            self.dismiss(self._settings)

    def action_go_back(self) -> None:
        self.dismiss(self._settings)

    # ── Helpers ─────────────────────────────────────────

    def _save_and_close(self) -> None:
        """Read every field and update the settings object."""
        radio = self.query_one("#backend-radio", RadioSet)
        self._settings.backend = self._index_to_backend(radio.pressed_index)
        self._settings.model = (
            self.query_one("#model-input", Input).value.strip()
        )
        self._settings.server_url = (
            self.query_one("#url-input", Input).value.strip()
        )
        self._settings.output_file = (
            self.query_one("#output-input", Input).value.strip()
            or DEFAULT_OUTPUT_FILE
        )
        self._settings.verbose = self.query_one(
            "#verbose-check", Checkbox
        ).value
        self._settings.batch_file = (
            self.query_one("#batch-file-input", Input).value.strip()
        )

        self.notify("Settings saved", title="✓")
        self.dismiss(self._settings)

    def _reset_defaults(self) -> None:
        """Reset all fields to factory defaults."""
        # backend → Ollama (set the first RadioButton's value to True)
        self.query_one("#rb-ollama", RadioButton).value = True

        self.query_one("#model-input", Input).value = ""
        self.query_one("#url-input", Input).value = ""
        self.query_one("#output-input", Input).value = DEFAULT_OUTPUT_FILE
        self.query_one("#verbose-check", Checkbox).value = False
        self.query_one("#batch-file-input", Input).value = ""

        cfg = BACKEND_DEFAULTS[Backend.OLLAMA]
        self.query_one("#model-input", Input).placeholder = cfg.default_model
        self.query_one("#url-input", Input).placeholder = cfg.default_url

        self.notify("Reset to defaults", title="↻")

    @staticmethod
    def _index_to_backend(index: int) -> Backend:
        return [Backend.OLLAMA, Backend.MLX, Backend.EXTERNAL][index]
