"""GPT/Codex-only credential-pool auth subcommands."""

from __future__ import annotations

from getpass import getpass
import math
import sys
from types import SimpleNamespace
import time
import uuid

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CUSTOM_POOL_PREFIX,
    SOURCE_MANUAL,
    STATUS_EXHAUSTED,
    STRATEGY_FILL_FIRST,
    STRATEGY_LEAST_USED,
    STRATEGY_RANDOM,
    STRATEGY_ROUND_ROBIN,
    PooledCredential,
    _exhausted_until,
    _normalize_custom_pool_name,
    get_pool_strategy,
    label_from_token,
    list_custom_pool_providers,
    load_pool,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import PROVIDER_REGISTRY


_OAUTH_CAPABLE_PROVIDERS = {"openai-codex"}


def _get_custom_provider_names() -> list[tuple[str, str, str]]:
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        config = load_config()
    except Exception:
        return []
    result = []
    for entry in get_compatible_custom_providers(config):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        pool_key = f"{CUSTOM_POOL_PREFIX}{_normalize_custom_pool_name(name)}"
        provider_key = str(entry.get("provider_key", "") or "").strip()
        result.append((name.strip(), pool_key, provider_key))
    return result


def _resolve_custom_provider_input(raw: str) -> str | None:
    normalized = (raw or "").strip().lower().replace(" ", "-")
    if not normalized:
        return None
    if normalized.startswith(CUSTOM_POOL_PREFIX):
        return normalized
    for display_name, pool_key, provider_key in _get_custom_provider_names():
        if _normalize_custom_pool_name(display_name) == normalized:
            return pool_key
        if provider_key and provider_key.strip().lower() == normalized:
            return pool_key
    return None


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    custom_key = _resolve_custom_provider_input(normalized)
    if custom_key:
        return custom_key
    aliases = {
        "gpt": "openai",
        "openai-api": "openai",
        "or": "openrouter",
        "open-router": "openrouter",
        "aigateway": "ai-gateway",
        "vercel": "ai-gateway",
        "vercel-ai-gateway": "ai-gateway",
        "codex": "openai-codex",
        "chatgpt": "openai-codex",
    }
    return aliases.get(normalized, normalized)


def _provider_base_url(provider: str) -> str:
    if provider.startswith(CUSTOM_POOL_PREFIX):
        from agent.credential_pool import _get_custom_provider_config

        cp_config = _get_custom_provider_config(provider)
        if cp_config:
            return str(cp_config.get("base_url") or "").strip()
        return ""
    pconfig = PROVIDER_REGISTRY.get(provider)
    return pconfig.inference_base_url if pconfig else ""


def _oauth_default_label(provider: str, count: int) -> str:
    return f"{provider}-oauth-{count}"


def _api_key_default_label(count: int) -> str:
    return f"api-key-{count}"


def _display_source(source: str) -> str:
    return source.split(":", 1)[1] if source.startswith("manual:") else source


def _classify_exhausted_status(entry) -> tuple[str, bool]:
    code = getattr(entry, "last_error_code", None)
    reason = str(getattr(entry, "last_error_reason", "") or "").strip().lower()
    message = str(getattr(entry, "last_error_message", "") or "").strip().lower()

    if code == 429 or any(token in reason for token in ("rate_limit", "usage_limit", "quota", "exhausted")) or any(
        token in message for token in ("rate limit", "usage limit", "quota", "too many requests")
    ):
        return "rate-limited", True

    if code in {401, 403} or any(token in reason for token in ("invalid_token", "invalid_grant", "unauthorized", "forbidden", "auth")) or any(
        token in message for token in ("unauthorized", "forbidden", "expired", "revoked", "invalid token", "authentication")
    ):
        return "auth failed", False

    return "exhausted", True


def _format_exhausted_status(entry) -> str:
    if entry.last_status != STATUS_EXHAUSTED:
        return ""
    label, show_retry_window = _classify_exhausted_status(entry)
    reason = getattr(entry, "last_error_reason", None)
    reason_text = f" {reason}" if isinstance(reason, str) and reason.strip() else ""
    code = f" ({entry.last_error_code})" if entry.last_error_code else ""
    if not show_retry_window:
        return f" {label}{reason_text}{code} (re-auth may be required)"
    exhausted_until = _exhausted_until(entry)
    if exhausted_until is None:
        return f" {label}{reason_text}{code}"
    remaining = max(0, int(math.ceil(exhausted_until - time.time())))
    if remaining <= 0:
        return f" {label}{reason_text}{code} (ready to retry)"
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        wait = f"{days}d {hours}h"
    elif hours:
        wait = f"{hours}h {minutes}m"
    elif minutes:
        wait = f"{minutes}m {seconds}s"
    else:
        wait = f"{seconds}s"
    return f" {label}{reason_text}{code} ({wait} left)"


def _ensure_supported_provider(provider: str) -> None:
    if provider in PROVIDER_REGISTRY or provider.startswith(CUSTOM_POOL_PREFIX):
        return
    raise SystemExit(
        "Unknown provider. Hermes Simple supports openai, openrouter, "
        "ai-gateway, openai-codex, and saved custom OpenAI-compatible endpoints."
    )


def auth_add_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    _ensure_supported_provider(provider)

    requested_type = str(getattr(args, "auth_type", "") or "").strip().lower()
    if requested_type in {AUTH_TYPE_API_KEY, "api-key"}:
        requested_type = AUTH_TYPE_API_KEY
    if not requested_type:
        requested_type = AUTH_TYPE_OAUTH if provider == "openai-codex" else AUTH_TYPE_API_KEY

    pool = load_pool(provider)

    if requested_type == AUTH_TYPE_API_KEY:
        token = (getattr(args, "api_key", None) or "").strip()
        if not token:
            token = getpass("Paste your API key: ").strip()
        if not token:
            raise SystemExit("No API key provided.")
        default_label = _api_key_default_label(len(pool.entries()) + 1)
        label = (getattr(args, "label", None) or "").strip()
        if not label:
            if sys.stdin.isatty():
                label = input(f"Label (optional, default: {default_label}): ").strip() or default_label
            else:
                label = default_label
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=token,
            base_url=_provider_base_url(provider),
        )
        pool.add_entry(entry)
        print(f'Added {provider} credential #{len(pool.entries())}: "{label}"')
        return

    if provider != "openai-codex":
        raise SystemExit("OAuth is only supported for openai-codex in Hermes Simple.")

    auth_mod.unsuppress_credential_source(provider, "device_code")
    creds = auth_mod._codex_device_code_login()
    label = (getattr(args, "label", None) or "").strip() or label_from_token(
        creds["tokens"]["access_token"],
        _oauth_default_label(provider, len(pool.entries()) + 1),
    )
    entry = PooledCredential(
        provider=provider,
        id=uuid.uuid4().hex[:6],
        label=label,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=f"{SOURCE_MANUAL}:device_code",
        access_token=creds["tokens"]["access_token"],
        refresh_token=creds["tokens"].get("refresh_token"),
        base_url=creds.get("base_url"),
        last_refresh=creds.get("last_refresh"),
    )
    pool.add_entry(entry)
    print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')


def auth_list_command(args) -> None:
    provider_filter = _normalize_provider(getattr(args, "provider", "") or "")
    providers = [provider_filter] if provider_filter else sorted({
        *PROVIDER_REGISTRY.keys(),
        *list_custom_pool_providers(),
    })
    for provider in providers:
        pool = load_pool(provider)
        entries = pool.entries()
        if not entries:
            continue
        current = pool.peek()
        print(f"{provider} ({len(entries)} credentials):")
        for idx, entry in enumerate(entries, start=1):
            marker = "  "
            if current is not None and entry.id == current.id:
                marker = "< "
            status = _format_exhausted_status(entry)
            source = _display_source(entry.source)
            print(f"  #{idx}  {entry.label:<20} {entry.auth_type:<7} {source}{status} {marker}".rstrip())
        print()


def auth_remove_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    _ensure_supported_provider(provider)
    target = getattr(args, "target", None)
    if target is None:
        target = getattr(args, "index", None)
    pool = load_pool(provider)
    index, matched, error = pool.resolve_target(target)
    if matched is None or index is None:
        raise SystemExit(f"{error} Provider: {provider}.")
    removed = pool.remove_index(index)
    if removed is None:
        raise SystemExit(f'No credential matching "{target}" for provider {provider}.')
    print(f"Removed {provider} credential #{index} ({removed.label})")


def auth_reset_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    _ensure_supported_provider(provider)
    pool = load_pool(provider)
    count = pool.reset_statuses()
    print(f"Reset status on {count} {provider} credentials")


def auth_status_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", "") or "")
    if not provider:
        raise SystemExit("Provider is required. Example: `hermes auth status openai-codex`.")
    _ensure_supported_provider(provider)
    status = auth_mod.get_auth_status(provider)
    if not status.get("logged_in"):
        reason = status.get("error")
        print(f"{provider}: logged out" + (f" ({reason})" if reason else ""))
        return
    print(f"{provider}: logged in")
    for key in ("auth_type", "expires_at", "api_base_url", "base_url"):
        value = status.get(key)
        if value:
            print(f"  {key}: {value}")


def auth_logout_command(args) -> None:
    auth_mod.logout_command(SimpleNamespace(provider=getattr(args, "provider", None)))


def _interactive_auth() -> None:
    print("Credential Pool Status")
    print("=" * 50)
    auth_list_command(SimpleNamespace(provider=None))
    print()

    choices = [
        "Add a credential",
        "Remove a credential",
        "Reset cooldowns for a provider",
        "Set rotation strategy for a provider",
        "Exit",
    ]
    print("What would you like to do?")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")

    try:
        raw = input("\nChoice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw or raw == str(len(choices)):
        return
    if raw == "1":
        _interactive_add()
    elif raw == "2":
        _interactive_remove()
    elif raw == "3":
        _interactive_reset()
    elif raw == "4":
        _interactive_strategy()


def _pick_provider(prompt: str = "Provider") -> str:
    known = sorted(PROVIDER_REGISTRY.keys())
    custom_names = _get_custom_provider_names()
    print(f"\nKnown providers: {', '.join(known)}")
    if custom_names:
        print(f"Custom endpoints: {', '.join(name for name, _key, _provider_key in custom_names)}")
    try:
        raw = input(f"{prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit()
    return _normalize_provider(raw)


def _interactive_add() -> None:
    provider = _pick_provider("Provider to add credential for")
    _ensure_supported_provider(provider)
    if provider in _OAUTH_CAPABLE_PROVIDERS:
        print(f"\n{provider} supports API keys and OAuth login.")
        print("  1. API key")
        print("  2. OAuth login")
        try:
            type_choice = input("Type [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        auth_type = AUTH_TYPE_OAUTH if type_choice == "2" else AUTH_TYPE_API_KEY
    else:
        auth_type = AUTH_TYPE_API_KEY

    label = None
    try:
        typed_label = input("Label / account name (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if typed_label:
        label = typed_label

    auth_add_command(SimpleNamespace(provider=provider, auth_type=auth_type, label=label, api_key=None))


def _interactive_remove() -> None:
    provider = _pick_provider("Provider to remove credential from")
    _ensure_supported_provider(provider)
    pool = load_pool(provider)
    if not pool.has_credentials():
        print(f"No credentials for {provider}.")
        return
    for i, entry in enumerate(pool.entries(), 1):
        exhausted = _format_exhausted_status(entry)
        print(f"  #{i}  {entry.label:25s} {entry.auth_type:10s} {entry.source}{exhausted} [id:{entry.id}]")
    try:
        raw = input("Remove #, id, or label (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if raw:
        auth_remove_command(SimpleNamespace(provider=provider, target=raw))


def _interactive_reset() -> None:
    provider = _pick_provider("Provider to reset cooldowns for")
    auth_reset_command(SimpleNamespace(provider=provider))


def _interactive_strategy() -> None:
    provider = _pick_provider("Provider to set strategy for")
    _ensure_supported_provider(provider)
    current = get_pool_strategy(provider)
    strategies = [STRATEGY_FILL_FIRST, STRATEGY_ROUND_ROBIN, STRATEGY_LEAST_USED, STRATEGY_RANDOM]
    print(f"\nCurrent strategy for {provider}: {current}")
    descriptions = {
        STRATEGY_FILL_FIRST: "Use first key until exhausted, then next",
        STRATEGY_ROUND_ROBIN: "Cycle through keys evenly",
        STRATEGY_LEAST_USED: "Always pick the least-used key",
        STRATEGY_RANDOM: "Random selection",
    }
    for i, strategy in enumerate(strategies, 1):
        marker = " <" if strategy == current else ""
        print(f"  {i}. {strategy:15s} - {descriptions.get(strategy, '')}{marker}")
    try:
        raw = input("\nStrategy [1-4]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return
    try:
        strategy = strategies[int(raw) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    pool_strategies = cfg.get("credential_pool_strategies") or {}
    if not isinstance(pool_strategies, dict):
        pool_strategies = {}
    pool_strategies[provider] = strategy
    cfg["credential_pool_strategies"] = pool_strategies
    save_config(cfg)
    print(f"Set {provider} strategy to: {strategy}")


def auth_command(args) -> None:
    action = getattr(args, "auth_action", "")
    if action == "add":
        auth_add_command(args)
        return
    if action == "list":
        auth_list_command(args)
        return
    if action == "remove":
        auth_remove_command(args)
        return
    if action == "reset":
        auth_reset_command(args)
        return
    if action == "status":
        auth_status_command(args)
        return
    if action == "logout":
        auth_logout_command(args)
        return
    _interactive_auth()
