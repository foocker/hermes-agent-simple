"""Runtime provider resolution for Hermes Simple.

Only GPT/Codex-compatible paths are supported:

- ``openai``: OpenAI API key via ``OPENAI_API_KEY``
- ``openrouter`` / ``ai-gateway``: third-party GPT API aggregators
- ``openai-codex``: Codex OAuth token from ``hermes_cli.auth``
- ``custom`` / named custom providers: OpenAI-compatible endpoints
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agent.credential_pool import get_custom_provider_pool_key, load_pool
from hermes_cli.auth import (
    AuthError,
    DEFAULT_AI_GATEWAY_BASE_URL,
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    has_usable_secret,
    resolve_api_key_provider_credentials,
    resolve_codex_runtime_credentials,
    resolve_provider,
)
from hermes_cli.config import get_compatible_custom_providers, load_config
from utils import base_url_hostname


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    hostname = base_url_hostname(base_url)
    if hostname == "api.openai.com":
        return "codex_responses"
    return None


def _parse_api_mode(raw: Any) -> Optional[str]:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"chat_completions", "codex_responses"}:
            return normalized
    return None


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        cfg = dict(model_cfg)
        if not cfg.get("default") and cfg.get("model"):
            cfg["default"] = cfg["model"]
        return cfg
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    if requested and requested.strip():
        return requested.strip().lower()
    model_cfg = _get_model_config()
    cfg_provider = model_cfg.get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()
    env_provider = os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider
    return "auto"


def _get_named_custom_provider(requested_provider: str) -> Optional[Dict[str, Any]]:
    requested_norm = _normalize_custom_provider_name(requested_provider or "")
    if not requested_norm or requested_norm in {"auto", "custom"}:
        return None
    if requested_norm.startswith("custom:"):
        requested_norm = requested_norm.split(":", 1)[1]

    config = load_config()
    providers = config.get("providers")
    if isinstance(providers, dict):
        for key, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            candidates = {_normalize_custom_provider_name(str(key))}
            display = entry.get("name")
            if isinstance(display, str) and display.strip():
                candidates.add(_normalize_custom_provider_name(display))
            if requested_norm not in candidates:
                continue
            base_url = entry.get("api") or entry.get("url") or entry.get("base_url") or ""
            if not isinstance(base_url, str) or not base_url.strip():
                continue
            key_env = str(entry.get("key_env", "") or "").strip()
            result = {
                "name": str(display or key),
                "base_url": base_url.strip(),
                "api_key": str(entry.get("api_key", "") or "").strip(),
                "key_env": key_env,
                "model": str(entry.get("default_model") or entry.get("model") or "").strip(),
            }
            api_mode = _parse_api_mode(entry.get("api_mode"))
            if api_mode:
                result["api_mode"] = api_mode
            return result

    for entry in get_compatible_custom_providers(config):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        base_url = entry.get("base_url")
        if not isinstance(name, str) or not isinstance(base_url, str):
            continue
        provider_key = str(entry.get("provider_key", "") or "").strip()
        candidates = {_normalize_custom_provider_name(name)}
        if provider_key:
            candidates.add(_normalize_custom_provider_name(provider_key))
        if requested_norm not in candidates:
            continue
        result = {
            "name": name.strip(),
            "base_url": base_url.strip(),
            "api_key": str(entry.get("api_key", "") or "").strip(),
            "key_env": str(entry.get("key_env", "") or "").strip(),
            "model": str(entry.get("model", "") or "").strip(),
        }
        api_mode = _parse_api_mode(entry.get("api_mode"))
        if api_mode:
            result["api_mode"] = api_mode
        return result
    return None


def _try_resolve_from_custom_pool(
    base_url: str,
    provider_label: str,
    api_mode_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    pool_key = get_custom_provider_pool_key(base_url)
    if not pool_key:
        return None
    try:
        pool = load_pool(pool_key)
        if not pool.has_credentials():
            return None
        entry = pool.select()
        if entry is None:
            return None
        api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        if not api_key:
            return None
        return {
            "provider": provider_label,
            "api_mode": api_mode_override or _detect_api_mode_for_url(base_url) or "chat_completions",
            "base_url": base_url,
            "api_key": api_key,
            "source": f"pool:{pool_key}",
            "credential_pool": pool,
        }
    except Exception:
        return None


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    requested_norm = (requested_provider or "").strip().lower()
    if requested_norm == "custom" and explicit_base_url:
        base_url = explicit_base_url.strip().rstrip("/")
        api_key = (
            (explicit_api_key or "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
            or "no-key-required"
        )
        return {
            "provider": "custom",
            "api_mode": _detect_api_mode_for_url(base_url) or "chat_completions",
            "base_url": base_url,
            "api_key": api_key,
            "source": "direct",
            "requested_provider": requested_provider,
        }

    custom_provider = _get_named_custom_provider(requested_provider)
    if not custom_provider:
        return None
    base_url = ((explicit_base_url or "").strip() or custom_provider.get("base_url", "")).rstrip("/")
    if not base_url:
        return None

    pool_result = _try_resolve_from_custom_pool(base_url, "custom", custom_provider.get("api_mode"))
    if pool_result:
        model_name = custom_provider.get("model")
        if model_name:
            pool_result["model"] = model_name
        pool_result["requested_provider"] = requested_provider
        return pool_result

    key_env = str(custom_provider.get("key_env", "") or "").strip()
    api_key_candidates = [
        (explicit_api_key or "").strip(),
        str(custom_provider.get("api_key", "") or "").strip(),
        os.getenv(key_env, "").strip() if key_env else "",
        os.getenv("OPENAI_API_KEY", "").strip(),
    ]
    api_key = next((candidate for candidate in api_key_candidates if has_usable_secret(candidate)), "")
    result = {
        "provider": "custom",
        "api_mode": custom_provider.get("api_mode") or _detect_api_mode_for_url(base_url) or "chat_completions",
        "base_url": base_url,
        "api_key": api_key or "no-key-required",
        "source": f"custom:{custom_provider.get('name', requested_provider)}",
        "requested_provider": requested_provider,
    }
    if custom_provider.get("model"):
        result["model"] = custom_provider["model"]
    return result


def _resolve_openai_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = _get_model_config()
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    cfg_api_key = ""
    for key in ("api_key", "api"):
        value = model_cfg.get(key)
        if isinstance(value, str) and value.strip():
            cfg_api_key = value.strip()
            break

    if requested_provider == "custom":
        base_url = (
            (explicit_base_url or "").strip()
            or (cfg_base_url.strip() if cfg_base_url else "")
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        provider = "custom" if base_url != DEFAULT_OPENAI_BASE_URL else "openai"
    else:
        base_url = (
            (explicit_base_url or "").strip()
            or (cfg_base_url.strip() if cfg_provider in {"openai", "auto", ""} else "")
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        provider = "openai" if base_url == DEFAULT_OPENAI_BASE_URL else "custom"

    api_key_candidates = [
        explicit_api_key,
        cfg_api_key,
        os.getenv("OPENAI_API_KEY"),
    ]
    api_key = next((str(candidate or "").strip() for candidate in api_key_candidates if has_usable_secret(candidate)), "")
    if not api_key and base_url == DEFAULT_OPENAI_BASE_URL:
        raise AuthError("OPENAI_API_KEY is not configured.", provider="openai", code="missing_api_key")

    return {
        "provider": provider,
        "api_mode": _detect_api_mode_for_url(base_url) or "chat_completions",
        "base_url": base_url,
        "api_key": api_key or "no-key-required",
        "source": "openai",
        "requested_provider": requested_provider,
    }


def _resolve_codex_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    if explicit_api_key:
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": (explicit_base_url or DEFAULT_CODEX_BASE_URL).rstrip("/"),
            "api_key": explicit_api_key,
            "source": "explicit",
            "requested_provider": requested_provider,
        }
    creds = resolve_codex_runtime_credentials()
    return {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": (explicit_base_url or creds.get("base_url") or DEFAULT_CODEX_BASE_URL).rstrip("/"),
        "api_key": creds.get("api_key", ""),
        "source": creds.get("source", "codex-auth"),
        "last_refresh": creds.get("last_refresh"),
        "requested_provider": requested_provider,
    }


def _resolve_gpt_aggregator_runtime(
    provider: str,
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    defaults = {
        "openrouter": DEFAULT_OPENROUTER_BASE_URL,
        "ai-gateway": DEFAULT_AI_GATEWAY_BASE_URL,
    }
    creds = resolve_api_key_provider_credentials(provider)
    base_url = (
        (explicit_base_url or "").strip()
        or str(creds.get("base_url") or "").strip()
        or defaults[provider]
    ).rstrip("/")
    api_key = (explicit_api_key or "").strip() or str(creds.get("api_key") or "").strip()
    if not api_key:
        raise AuthError(f"{provider} API key is not configured.", provider=provider, code="missing_api_key")
    return {
        "provider": provider,
        "api_mode": "chat_completions",
        "base_url": base_url,
        "api_key": api_key,
        "source": creds.get("source", provider),
        "requested_provider": requested_provider,
    }


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    del target_model
    requested_provider = resolve_requested_provider(requested)

    custom_runtime = _resolve_named_custom_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    if custom_runtime:
        return custom_runtime

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    if provider == "openai-codex":
        return _resolve_codex_runtime(
            requested_provider=requested_provider,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )
    if provider in {"openrouter", "ai-gateway"}:
        return _resolve_gpt_aggregator_runtime(
            provider,
            requested_provider=requested_provider,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )
    if provider in {"openai", "custom"}:
        return _resolve_openai_runtime(
            requested_provider=provider,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )
    raise AuthError(f"Provider '{provider}' was removed in Hermes Simple.", provider=provider, code="provider_removed")


def provider_error_hint(error: Exception, provider: str) -> str:
    if isinstance(error, AuthError):
        return str(error)
    return f"{provider}: {error}"
