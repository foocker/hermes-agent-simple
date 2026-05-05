"""GPT/Codex model catalogs and validation helpers for Hermes Simple."""

from __future__ import annotations

import json
import os
import urllib.request
from difflib import get_close_matches
from typing import Any, NamedTuple, Optional

from hermes_cli import __version__ as _HERMES_VERSION

_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"


OPENAI_MODELS: list[str] = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("openai/gpt-5.5", "recommended"),
    ("openai/gpt-5.5-pro", ""),
    ("openai/gpt-5.4", ""),
    ("openai/gpt-5.4-mini", ""),
    ("openai/gpt-5.3-codex", ""),
]

VERCEL_AI_GATEWAY_MODELS: list[tuple[str, str]] = [
    ("openai/gpt-5.4", "recommended"),
    ("openai/gpt-5.4-mini", ""),
    ("openai/gpt-5.3-codex", ""),
]

_openrouter_catalog_cache: list[tuple[str, str]] | None = None
_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}


def _codex_curated_models() -> list[str]:
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _add_forward_compat_models

    return _add_forward_compat_models(list(DEFAULT_CODEX_MODELS))


_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": OPENAI_MODELS,
    "openai-codex": _codex_curated_models(),
    "openrouter": [mid for mid, _ in OPENROUTER_MODELS],
    "ai-gateway": [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS],
}


class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str


CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("openai", "OpenAI GPT", "OpenAI GPT (OPENAI_API_KEY)"),
    ProviderEntry("openrouter", "OpenRouter", "OpenRouter GPT models"),
    ProviderEntry("ai-gateway", "Vercel AI Gateway", "Vercel AI Gateway GPT models"),
    ProviderEntry("openai-codex", "OpenAI Codex", "OpenAI Codex"),
]

_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"

_PROVIDER_ALIASES = {
    "gpt": "openai",
    "openai-api": "openai",
    "or": "openrouter",
    "open-router": "openrouter",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "codex": "openai-codex",
    "chatgpt": "openai-codex",
    "custom": "custom",
    "local": "custom",
    "ollama": "custom",
    "lmstudio": "custom",
    "lm-studio": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    "llama.cpp": "custom",
}

_KNOWN_PROVIDER_NAMES = set(_PROVIDER_LABELS) | set(_PROVIDER_ALIASES) | {"custom"}

# Kept for callers that still import these names while the simplified tree is
# being trimmed. They intentionally do not expand provider discovery.
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset()


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    del provider
    return list(curated)


def normalize_provider(provider: Optional[str]) -> str:
    normalized = (provider or "openai").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    original = (provider or "openai").strip()
    normalized = normalize_provider(original)
    if normalized == "auto":
        return "Auto"
    return _PROVIDER_LABELS.get(normalized, original or "OpenAI")


def get_default_model_for_provider(provider: str) -> str:
    models = provider_model_ids(provider)
    return models[0] if models else ""


def _is_gpt_model_id(model_id: str) -> bool:
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        vendor, raw = raw.split("/", 1)
        if vendor != "openai":
            return False
    return raw.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))


def _model_supports_tools(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    params = item.get("supported_parameters")
    if not isinstance(params, list):
        return True
    return "tools" in params


def fetch_openrouter_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    global _openrouter_catalog_cache
    if _openrouter_catalog_cache is not None and not force_refresh:
        return list(_openrouter_catalog_cache)

    fallback = [(mid, desc) for mid, desc in OPENROUTER_MODELS if mid.startswith("openai/")]
    preferred_ids = [mid for mid, _ in fallback]

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_openrouter_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_openrouter_catalog_cache or fallback)

    live_by_id = {
        str(item.get("id") or "").strip(): item
        for item in live_items
        if isinstance(item, dict) and str(item.get("id") or "").strip().startswith("openai/")
    }
    curated: list[tuple[str, str]] = []
    for preferred_id in preferred_ids:
        item = live_by_id.get(preferred_id)
        if item is None or not _model_supports_tools(item):
            continue
        pricing = item.get("pricing") if isinstance(item, dict) else None
        desc = ""
        if isinstance(pricing, dict):
            try:
                if float(pricing.get("prompt", "1")) == 0 and float(pricing.get("completion", "1")) == 0:
                    desc = "free"
            except (TypeError, ValueError):
                pass
        curated.append((preferred_id, desc))

    if not curated:
        return list(_openrouter_catalog_cache or fallback)
    curated[0] = (curated[0][0], "recommended")
    _openrouter_catalog_cache = curated
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def fetch_ai_gateway_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    global _ai_gateway_catalog_cache
    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from hermes_constants import AI_GATEWAY_BASE_URL

    fallback = [(mid, desc) for mid, desc in VERCEL_AI_GATEWAY_MODELS if mid.startswith("openai/")]
    preferred_ids = [mid for mid, _ in fallback]
    try:
        req = urllib.request.Request(
            f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models",
            headers={"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_ai_gateway_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_ai_gateway_catalog_cache or fallback)

    live = {
        str(item.get("id") or "").strip()
        for item in live_items
        if isinstance(item, dict) and str(item.get("id") or "").strip().startswith("openai/")
    }
    curated = [(mid, desc) for mid, desc in fallback if mid in live]
    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)
    curated[0] = (curated[0][0], "recommended")
    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]


def _format_price_per_mtok(per_token_str: str) -> str:
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    return f"${val * 1_000_000:.2f}"


def format_model_pricing_table(
    models: list[tuple[str, str]],
    pricing_map: dict[str, dict[str, str]],
    current_model: str = "",
    indent: str = "      ",
) -> list[str]:
    if not models:
        return []
    lines = [f"{indent}Model{' ' * 24} In       Out /Mtok"]
    for mid, _desc in models:
        p = pricing_map.get(mid, {})
        inp = _format_price_per_mtok(p.get("prompt", "")) if p else ""
        out = _format_price_per_mtok(p.get("completion", "")) if p else ""
        marker = "  <- current" if mid == current_model else ""
        lines.append(f"{indent}{mid:<28} {inp:>8} {out:>8}{marker}")
    return lines


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    cache_key = (base_url or "").rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    headers = {"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(cache_key.rstrip("/") + "/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        pricing = item.get("pricing")
        if mid.startswith("openai/") and isinstance(pricing, dict):
            result[mid] = {
                "prompt": str(pricing.get("prompt", "")),
                "completion": str(pricing.get("completion", "")),
            }
    _pricing_cache[cache_key] = result
    return result


def fetch_ai_gateway_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    from hermes_constants import AI_GATEWAY_BASE_URL

    cache_key = AI_GATEWAY_BASE_URL.rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]
    try:
        req = urllib.request.Request(
            f"{cache_key}/models",
            headers={"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        pricing = item.get("pricing")
        if mid.startswith("openai/") and isinstance(pricing, dict):
            result[mid] = {
                "prompt": str(pricing.get("input", "")),
                "completion": str(pricing.get("output", "")),
            }
    _pricing_cache[cache_key] = result
    return result


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_models_with_pricing(
            api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            base_url="https://openrouter.ai/api",
            force_refresh=force_refresh,
        )
    if normalized == "ai-gateway":
        return fetch_ai_gateway_pricing(force_refresh=force_refresh)
    return {}


def list_available_providers() -> list[dict[str, Any]]:
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)
    result: list[dict[str, Any]] = []
    for entry in CANONICAL_PROVIDERS:
        has_creds = False
        if entry.slug == "openai":
            has_creds = bool(os.getenv("OPENAI_API_KEY", ""))
        elif entry.slug == "openrouter":
            has_creds = bool(os.getenv("OPENROUTER_API_KEY", ""))
        elif entry.slug == "ai-gateway":
            has_creds = bool(os.getenv("AI_GATEWAY_API_KEY", ""))
        elif entry.slug == "openai-codex":
            try:
                from hermes_cli.auth import get_codex_auth_status

                has_creds = bool(get_codex_auth_status().get("logged_in"))
            except Exception:
                has_creds = False
        result.append({
            "id": entry.slug,
            "label": entry.label,
            "aliases": aliases_for.get(entry.slug, []),
            "authenticated": has_creds,
        })
    result.append({"id": "custom", "label": "Custom endpoint", "aliases": aliases_for.get("custom", []), "authenticated": bool(_get_custom_base_url())})
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            if provider_part == "custom" and ":" in model_part:
                custom_name, _, actual_model = model_part.partition(":")
                if custom_name and actual_model:
                    return (f"custom:{custom_name}", actual_model.strip())
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    try:
        from hermes_cli.config import load_config

        model_cfg = load_config().get("model", {})
        if isinstance(model_cfg, dict):
            return str(model_cfg.get("base_url", "")).strip()
    except Exception:
        pass
    return ""


def curated_models_for_provider(provider: Optional[str], *, force_refresh: bool = False) -> list[tuple[str, str]]:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_openrouter_models(force_refresh=force_refresh)
    if normalized == "ai-gateway":
        return fetch_ai_gateway_models(force_refresh=force_refresh)
    return [(m, "") for m in provider_model_ids(normalized, force_refresh=force_refresh)]


def detect_static_provider_for_model(model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    name = (model_name or "").strip()
    if not name:
        return None
    lower = name.lower()
    current = normalize_provider(current_provider)
    for provider, models in _PROVIDER_MODELS.items():
        if provider == current:
            continue
        if any(lower == m.lower() for m in models):
            return provider, name
    return None


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    lower = model_name.strip().lower()
    if not lower:
        return None
    for mid in model_ids():
        if lower == mid.lower():
            return mid
        if "/" in mid and lower == mid.split("/", 1)[1].lower():
            return mid
    return None


def detect_provider_for_model(model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    static = detect_static_provider_for_model(model_name, current_provider)
    if static:
        return static
    or_slug = _find_openrouter_slug(model_name)
    if or_slug and normalize_provider(current_provider) != "openrouter":
        return "openrouter", or_slug
    return None


def _strip_vendor_prefix(model_id: str) -> str:
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    base = _strip_vendor_prefix(str(model_id or "")).split(":", 1)[0]
    return bool(base and "codex" not in base and base.startswith(("gpt-", "o1", "o3", "o4")))


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    return _is_openai_fast_model(model_id)


def resolve_fast_mode_overrides(model_id: Optional[str]) -> dict[str, Any] | None:
    if not model_supports_fast_mode(model_id):
        return None
    return {"service_tier": "priority"}


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return model_ids(force_refresh=force_refresh)
    if normalized == "ai-gateway":
        return ai_gateway_model_ids(force_refresh=force_refresh)
    if normalized == "openai-codex":
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials
            from hermes_cli.codex_models import get_codex_model_ids

            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
            return get_codex_model_ids(access_token=creds.get("api_key"))
        except Exception:
            return list(_PROVIDER_MODELS["openai-codex"])
    if normalized == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base = (os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1").rstrip("/")
        if api_key:
            live = fetch_api_models(api_key, base)
            if live:
                filtered = [mid for mid in live if _is_gpt_model_id(mid)]
                if filtered:
                    return filtered
        return list(_PROVIDER_MODELS["openai"])
    if normalized == "custom":
        base = _get_custom_base_url()
        if base:
            live = fetch_api_models(os.getenv("OPENAI_API_KEY", ""), base)
            if live:
                return live
    return list(_PROVIDER_MODELS.get(normalized, []))


def probe_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    del api_mode
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return {"models": None, "probed_url": None, "resolved_base_url": "", "suggested_base_url": None, "used_fallback": False}

    alternate = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates = [(normalized, False)]
    if alternate and alternate != normalized:
        candidates.append((alternate, True))

    headers = {"User-Agent": _HERMES_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    tried: list[str] = []
    for candidate, used_fallback in candidates:
        url = candidate.rstrip("/") + "/models"
        tried.append(url)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            data = payload.get("data", []) if isinstance(payload, dict) else []
            models = [str(item.get("id") or "").strip() for item in data if isinstance(item, dict) and item.get("id")]
            return {
                "models": models,
                "probed_url": url,
                "resolved_base_url": candidate.rstrip("/"),
                "suggested_base_url": alternate if alternate != candidate else normalized,
                "used_fallback": used_fallback,
            }
        except Exception:
            continue
    return {
        "models": None,
        "probed_url": tried[0] if tried else normalized.rstrip("/") + "/models",
        "resolved_base_url": normalized,
        "suggested_base_url": alternate if alternate != normalized else None,
        "used_fallback": False,
    }


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
) -> Optional[list[str]]:
    return probe_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode).get("models")


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    del api_mode
    requested = (model_name or "").strip()
    normalized = normalize_provider(provider)
    if not requested:
        return {"accepted": False, "persist": False, "recognized": False, "message": "Model name cannot be empty."}
    if any(ch.isspace() for ch in requested):
        return {"accepted": False, "persist": False, "recognized": False, "message": "Model names cannot contain spaces."}

    if normalized == "custom":
        probe = probe_api_models(api_key, base_url)
        api_models = probe.get("models")
        if api_models is None:
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. Hermes will still save `{requested}`.",
            }
        if requested in set(api_models):
            return {"accepted": True, "persist": True, "recognized": True, "message": None}
        auto = get_close_matches(requested, api_models, n=1, cutoff=0.9)
        if auto:
            return {"accepted": True, "persist": True, "recognized": True, "corrected_model": auto[0], "message": f"Auto-corrected `{requested}` -> `{auto[0]}`"}
        suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
        suffix = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions) if suggestions else ""
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": f"Note: `{requested}` was not found in this custom endpoint's model listing ({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models.{suffix}",
        }

    catalog_models = provider_model_ids(normalized)
    catalog_lower = {m.lower(): m for m in catalog_models}
    if requested.lower() in catalog_lower:
        return {"accepted": True, "persist": True, "recognized": True, "corrected_model": catalog_lower[requested.lower()], "message": None}
    auto = get_close_matches(requested.lower(), list(catalog_lower), n=1, cutoff=0.9)
    if auto:
        corrected = catalog_lower[auto[0]]
        return {"accepted": True, "persist": True, "recognized": True, "corrected_model": corrected, "message": f"Auto-corrected `{requested}` -> `{corrected}`"}
    if normalized in _PROVIDER_MODELS:
        suggestions = get_close_matches(requested.lower(), list(catalog_lower), n=3, cutoff=0.5)
        suffix = "\n  Similar models: " + ", ".join(f"`{catalog_lower[s]}`" for s in suggestions) if suggestions else ""
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": f"Model `{requested}` was not found in the {provider_label(normalized)} model listing.{suffix}",
        }
    return {"accepted": True, "persist": True, "recognized": False, "message": None}
