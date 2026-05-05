"""Shared helpers for tool backend selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

_DEFAULT_BROWSER_PROVIDER = "local"
_DEFAULT_MODAL_MODE = "auto"
_VALID_MODAL_MODES = {"auto", "direct"}


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


def coerce_modal_mode(value: object | None) -> str:
    """Return the requested modal mode when valid, else the default."""
    mode = str(value or _DEFAULT_MODAL_MODE).strip().lower()
    if mode in _VALID_MODAL_MODES:
        return mode
    return _DEFAULT_MODAL_MODE


def normalize_modal_mode(value: object | None) -> str:
    """Return a normalized modal execution mode."""
    return coerce_modal_mode(value)


def has_direct_modal_credentials() -> bool:
    """Return True when direct Modal credentials/config are available."""
    return bool(
        (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
        or (Path.home() / ".modal.toml").exists()
    )


def resolve_modal_backend_state(
    modal_mode: object | None,
    *,
    has_direct: bool,
    managed_ready: bool,
) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection.

    Semantics:
    - ``direct`` means direct-only
    - ``auto`` uses direct credentials when available
    """
    _ = managed_ready
    requested_mode = coerce_modal_mode(modal_mode)
    normalized_mode = normalize_modal_mode(modal_mode)
    selected_backend = "direct" if has_direct else None

    return {
        "requested_mode": requested_mode,
        "mode": normalized_mode,
        "has_direct": has_direct,
        "managed_ready": False,
        "managed_mode_blocked": False,
        "selected_backend": selected_backend,
    }


def resolve_openai_audio_api_key() -> str:
    """Prefer the voice-tools key, but fall back to the normal OpenAI key."""
    return (
        os.getenv("VOICE_TOOLS_OPENAI_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()


def fal_key_is_configured() -> bool:
    """Return True when FAL_KEY is set to a non-whitespace value.

    Consults both ``os.environ`` and ``~/.hermes/.env`` (via
    ``hermes_cli.config.get_env_value`` when available) so tool-side
    checks and CLI setup-time checks agree.  A whitespace-only value
    is treated as unset everywhere.
    """
    value = os.getenv("FAL_KEY")
    if value is None:
        # Fall back to the .env file for CLI paths that may run before
        # dotenv is loaded into os.environ.
        try:
            from hermes_cli.config import get_env_value

            value = get_env_value("FAL_KEY")
        except Exception:
            value = None
    return bool(value and value.strip())
