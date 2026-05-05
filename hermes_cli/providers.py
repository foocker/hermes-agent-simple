"""Provider identity for Hermes Simple.

Built-ins are limited to OpenAI GPT and OpenAI Codex. User-configured
``providers:`` and ``custom_providers:`` entries are treated as
OpenAI-compatible custom endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)


# -- Hermes overlay ----------------------------------------------------------
# Hermes Simple built-in provider metadata.

@dataclass(frozen=True)
class HermesOverlay:
    """Hermes Simple provider metadata."""

    transport: str = "openai_chat"        # openai_chat | codex_responses
    is_aggregator: bool = False
    auth_type: str = "api_key"            # api_key | oauth_device_code | oauth_external | external_process
    extra_env_vars: Tuple[str, ...] = ()
    base_url_override: str = ""
    base_url_env_var: str = ""            # env var for user-custom base URL


HERMES_OVERLAYS: Dict[str, HermesOverlay] = {
    "openai": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("OPENAI_API_KEY",),
        base_url_override="https://api.openai.com/v1",
        base_url_env_var="OPENAI_BASE_URL",
    ),
    "openrouter": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        extra_env_vars=("OPENROUTER_API_KEY",),
        base_url_override="https://openrouter.ai/api/v1",
        base_url_env_var="OPENROUTER_BASE_URL",
    ),
    "ai-gateway": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        extra_env_vars=("AI_GATEWAY_API_KEY",),
        base_url_override="https://ai-gateway.vercel.sh/v1",
        base_url_env_var="AI_GATEWAY_BASE_URL",
    ),
    "openai-codex": HermesOverlay(
        transport="codex_responses",
        auth_type="oauth_external",
        base_url_override="https://chatgpt.com/backend-api/codex",
    ),
}


# -- Resolved provider -------------------------------------------------------
# The merged result of built-ins + user config.

@dataclass
class ProviderDef:
    """Complete provider definition — merged from all sources."""

    id: str
    name: str
    transport: str                        # openai_chat | codex_responses
    api_key_env_vars: Tuple[str, ...]     # all env vars to check for API key
    base_url: str = ""
    base_url_env_var: str = ""
    is_aggregator: bool = False
    auth_type: str = "api_key"
    doc: str = ""
    source: str = ""


# -- Aliases ------------------------------------------------------------------
# Maps human-friendly / legacy names to canonical provider IDs.
ALIASES: Dict[str, str] = {
    "gpt": "openai",
    "openai-api": "openai",
    "or": "openrouter",
    "open-router": "openrouter",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "codex": "openai-codex",
    "chatgpt": "openai-codex",
    "local": "custom",
    "lmstudio": "custom",
    "lm-studio": "custom",
    "lm_studio": "custom",
    "ollama": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    "llama.cpp": "custom",
    "llama-cpp": "custom",
}


# -- Display labels -----------------------------------------------------------
_LABEL_OVERRIDES: Dict[str, str] = {
    "openai": "OpenAI GPT",
    "openrouter": "OpenRouter",
    "ai-gateway": "Vercel AI Gateway",
    "openai-codex": "OpenAI Codex",
    "custom": "Custom endpoint",
}


# -- Transport → API mode mapping ---------------------------------------------

TRANSPORT_TO_API_MODE: Dict[str, str] = {
    "openai_chat": "chat_completions",
    "codex_responses": "codex_responses",
}


# -- Helper functions ---------------------------------------------------------

def normalize_provider(name: str) -> str:
    """Resolve aliases and normalise casing to a canonical provider id.

    Returns the canonical id string.  Does *not* validate that the id
    corresponds to a known provider.
    """
    key = name.strip().lower()
    return ALIASES.get(key, key)


def get_provider(name: str) -> Optional[ProviderDef]:
    """Look up a built-in provider by id or alias.

    Hermes Simple has no models.dev provider fan-out. Built-ins are only
    OpenAI GPT and OpenAI Codex; user-defined OpenAI-compatible providers are
    resolved by :func:`resolve_provider_full`.
    """
    canonical = normalize_provider(name)
    overlay = HERMES_OVERLAYS.get(canonical)
    if overlay is None:
        return None
    return ProviderDef(
        id=canonical,
        name=_LABEL_OVERRIDES.get(canonical, canonical),
        transport=overlay.transport,
        api_key_env_vars=overlay.extra_env_vars,
        base_url=overlay.base_url_override,
        base_url_env_var=overlay.base_url_env_var,
        is_aggregator=overlay.is_aggregator,
        auth_type=overlay.auth_type,
        source="hermes-simple",
    )


def get_label(provider_id: str) -> str:
    """Get a human-readable display name for a provider."""
    canonical = normalize_provider(provider_id)

    # Check label overrides first
    if canonical in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[canonical]

    pdef = get_provider(canonical)
    if pdef:
        return pdef.name

    return canonical




def is_aggregator(provider: str) -> bool:
    """Return True when the provider is a multi-model aggregator."""
    pdef = get_provider(provider)
    return pdef.is_aggregator if pdef else False


def determine_api_mode(provider: str, base_url: str = "") -> str:
    """Determine the API mode (wire protocol) for a provider/endpoint.

    Resolution order:
      1. Known provider → transport → TRANSPORT_TO_API_MODE.
      2. URL heuristics for unknown / custom providers.
      3. Default: 'chat_completions'.
    """
    pdef = get_provider(provider)
    if pdef is not None:
        if base_url and base_url_host_matches(base_url, "api.openai.com"):
            return "codex_responses"
        return TRANSPORT_TO_API_MODE.get(pdef.transport, "chat_completions")

    # URL-based heuristics for custom / unknown providers
    if base_url:
        hostname = base_url_hostname(base_url)
        if hostname == "api.openai.com":
            return "codex_responses"

    return "chat_completions"


# -- Provider from user config ------------------------------------------------

def resolve_user_provider(name: str, user_config: Dict[str, Any]) -> Optional[ProviderDef]:
    """Resolve a provider from the user's config.yaml ``providers:`` section.

    Args:
        name: Provider name as given by the user.
        user_config: The ``providers:`` dict from config.yaml.

    Returns:
        ProviderDef if found, else None.
    """
    if not user_config or not isinstance(user_config, dict):
        return None

    entry = user_config.get(name)
    if not isinstance(entry, dict):
        return None

    # Extract fields
    display_name = entry.get("name", "") or name
    api_url = entry.get("api", "") or entry.get("url", "") or entry.get("base_url", "") or ""
    key_env = entry.get("key_env", "") or ""
    transport = entry.get("transport", "openai_chat") or "openai_chat"

    env_vars: List[str] = []
    if key_env:
        env_vars.append(key_env)

    return ProviderDef(
        id=name,
        name=display_name,
        transport=transport,
        api_key_env_vars=tuple(env_vars),
        base_url=api_url,
        is_aggregator=False,
        auth_type="api_key",
        source="user-config",
    )


def custom_provider_slug(display_name: str) -> str:
    """Build a canonical slug for a custom_providers entry.

    Matches the convention used by runtime_provider and credential_pool
    (``custom:<normalized-name>``).  Centralised here so all call-sites
    produce identical slugs.
    """
    return "custom:" + display_name.strip().lower().replace(" ", "-")


def resolve_custom_provider(
    name: str,
    custom_providers: Optional[List[Dict[str, Any]]],
) -> Optional[ProviderDef]:
    """Resolve a provider from the user's config.yaml ``custom_providers`` list."""
    if not custom_providers or not isinstance(custom_providers, list):
        return None

    requested = (name or "").strip().lower()
    if not requested:
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue

        display_name = (entry.get("name") or "").strip()
        api_url = (
            entry.get("base_url", "")
            or entry.get("url", "")
            or entry.get("api", "")
            or ""
        ).strip()
        if not display_name or not api_url:
            continue

        slug = custom_provider_slug(display_name)
        if requested not in {display_name.lower(), slug}:
            continue

        return ProviderDef(
            id=slug,
            name=display_name,
            transport="openai_chat",
            api_key_env_vars=(),
            base_url=api_url,
            is_aggregator=False,
            auth_type="api_key",
            source="user-config",
        )

    return None


def resolve_provider_full(
    name: str,
    user_providers: Optional[Dict[str, Any]] = None,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
) -> Optional[ProviderDef]:
    """Full resolution chain: built-in → user config.

    This is the main entry point for --provider flag resolution.

    Args:
        name: Provider name or alias.
        user_providers: The ``providers:`` dict from config.yaml (optional).
        custom_providers: The ``custom_providers:`` list from config.yaml (optional).

    Returns:
        ProviderDef if found, else None.
    """
    canonical = normalize_provider(name)

    # 1. Built-in (models.dev + overlays)
    pdef = get_provider(canonical)
    if pdef is not None:
        return pdef

    # 2. User-defined providers from config
    if user_providers:
        # Try canonical name
        user_pdef = resolve_user_provider(canonical, user_providers)
        if user_pdef is not None:
            return user_pdef
        # Try original name (in case alias didn't match)
        user_pdef = resolve_user_provider(name.strip().lower(), user_providers)
        if user_pdef is not None:
            return user_pdef

    # 2b. Saved custom providers from config
    custom_pdef = resolve_custom_provider(name, custom_providers)
    if custom_pdef is not None:
        return custom_pdef

    return None
