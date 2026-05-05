"""GPT/Codex model-name normalization for Hermes Simple."""

from __future__ import annotations

from typing import Optional

_AGGREGATOR_PROVIDERS: frozenset[str] = frozenset({
    "openrouter",
    "ai-gateway",
})


def _strip_vendor_prefix(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _normalize_provider_alias(provider_name: str) -> str:
    raw = (provider_name or "").strip().lower()
    if not raw:
        return raw
    try:
        from hermes_cli.models import normalize_provider

        return normalize_provider(raw)
    except Exception:
        return raw


def detect_vendor(model_name: str) -> Optional[str]:
    name = (model_name or "").strip().lower()
    if not name:
        return None
    if "/" in name:
        vendor = name.split("/", 1)[0]
        return vendor or None
    if name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    return None


def normalize_model_for_provider(model_input: str, target_provider: str) -> str:
    name = (model_input or "").strip()
    if not name:
        return name

    provider = _normalize_provider_alias(target_provider)
    if provider in _AGGREGATOR_PROVIDERS:
        if "/" in name:
            return name
        vendor = detect_vendor(name)
        return f"{vendor}/{name}" if vendor else name

    if provider == "openai-codex":
        return _strip_vendor_prefix(name)

    return name


def normalize_models_for_provider(models: list[str], target_provider: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        normalized = normalize_model_for_provider(model, target_provider)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
