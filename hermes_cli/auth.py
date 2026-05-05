"""GPT/Codex-only authentication helpers for Hermes Simple.

The full Hermes tree supports many provider-specific auth flows.  Hermes Simple
keeps only:

- OpenAI API key for GPT models
- OpenAI Codex OAuth tokens
- user-defined OpenAI-compatible custom endpoints via the credential pool

Compatibility stubs remain for older call sites while they are being removed.
They deliberately do not register or auto-select non-GPT providers.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

from hermes_cli.config import get_config_path, get_env_value, get_hermes_home, read_raw_config
from utils import atomic_replace

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback
    fcntl = None
try:
    import msvcrt
except Exception:  # pragma: no cover - Unix fallback
    msvcrt = None


AUTH_STORE_VERSION = 1
AUTH_LOCK_TIMEOUT_SECONDS = 15.0

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

LMSTUDIO_NOAUTH_PLACEHOLDER = "dummy-lm-api-key"


@dataclass
class ProviderConfig:
    """Describes a supported inference provider."""

    id: str
    name: str
    auth_type: str
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    api_key_env_vars: tuple = ()
    base_url_env_var: str = ""


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        id="openai",
        name="OpenAI",
        auth_type="api_key",
        inference_base_url=DEFAULT_OPENAI_BASE_URL,
        api_key_env_vars=("OPENAI_API_KEY",),
        base_url_env_var="OPENAI_BASE_URL",
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        auth_type="api_key",
        inference_base_url=DEFAULT_OPENROUTER_BASE_URL,
        api_key_env_vars=("OPENROUTER_API_KEY",),
        base_url_env_var="OPENROUTER_BASE_URL",
    ),
    "ai-gateway": ProviderConfig(
        id="ai-gateway",
        name="Vercel AI Gateway",
        auth_type="api_key",
        inference_base_url=DEFAULT_AI_GATEWAY_BASE_URL,
        api_key_env_vars=("AI_GATEWAY_API_KEY",),
        base_url_env_var="AI_GATEWAY_BASE_URL",
    ),
    "openai-codex": ProviderConfig(
        id="openai-codex",
        name="OpenAI Codex",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_CODEX_BASE_URL,
    ),
}

SERVICE_PROVIDER_NAMES: Dict[str, str] = {}

_PLACEHOLDER_SECRET_VALUES = {
    "*",
    "**",
    "***",
    "changeme",
    "your_api_key",
    "your-api-key",
    "placeholder",
    "example",
    "dummy",
    "null",
    "none",
}


class AuthError(RuntimeError):
    """Structured auth error with UX mapping hints."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        code: Optional[str] = None,
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


def format_auth_error(error: Exception) -> str:
    if isinstance(error, AuthError) and error.relogin_required:
        return f"{error} Run `hermes auth add openai-codex` to re-authenticate."
    return str(error)


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if len(cleaned) < min_length:
        return False
    return cleaned.lower() not in _PLACEHOLDER_SECRET_VALUES


def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_auth = (Path.home() / ".hermes" / "auth.json").resolve(strict=False)
        if path.resolve(strict=False) == real_home_auth:
            raise RuntimeError(f"Refusing to touch real user auth store during test run: {path}")
    return path


def _auth_lock_path() -> Path:
    return _auth_file_path().with_suffix(".lock")


_auth_lock_holder = threading.local()


@contextmanager
def _auth_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-process advisory lock for auth.json reads/writes. Reentrant."""

    if getattr(_auth_lock_holder, "depth", 0) > 0:
        _auth_lock_holder.depth += 1
        try:
            yield
        finally:
            _auth_lock_holder.depth -= 1
        return

    lock_path = _auth_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        _auth_lock_holder.depth = 1
        try:
            yield
        finally:
            _auth_lock_holder.depth = 0
        return

    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    with lock_path.open("r+" if msvcrt else "a+") as lock_file:
        deadline = time.time() + max(1.0, timeout_seconds)
        while True:
            try:
                if fcntl:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except (BlockingIOError, OSError, PermissionError):
                if time.time() >= deadline:
                    raise TimeoutError("Timed out waiting for auth store lock")
                time.sleep(0.05)

        _auth_lock_holder.depth = 1
        try:
            yield
        finally:
            _auth_lock_holder.depth = 0
            if fcntl:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return {"version": AUTH_STORE_VERSION, "providers": {}}
    try:
        raw = json.loads(auth_file.read_text())
    except Exception as exc:
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        try:
            import shutil

            shutil.copy2(auth_file, corrupt_path)
        except Exception:
            pass
        logger.warning("auth: failed to parse %s (%s); preserved at %s", auth_file, exc, corrupt_path)
        return {"version": AUTH_STORE_VERSION, "providers": {}}
    if isinstance(raw, dict):
        raw.setdefault("providers", {})
        return raw
    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _save_auth_store(auth_store: Dict[str, Any]) -> Path:
    auth_file = _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(auth_store, indent=2) + "\n"
    tmp_path = auth_file.with_name(f"{auth_file.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(tmp_path, auth_file)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    try:
        auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return auth_file


def _load_provider_state(auth_store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    providers = auth_store.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(provider_id)
    return dict(state) if isinstance(state, dict) else None


def _save_provider_state(auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    auth_store["active_provider"] = provider_id


def _store_provider_state(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    *,
    set_active: bool = True,
) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    if set_active:
        auth_store["active_provider"] = provider_id


def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    pool = _load_auth_store().get("credential_pool")
    if not isinstance(pool, dict):
        pool = {}
    if provider_id is None:
        return dict(pool)
    entries = pool.get(provider_id)
    return list(entries) if isinstance(entries, list) else []


def write_credential_pool(provider_id: str, entries: List[Dict[str, Any]]) -> Path:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool
        pool[provider_id] = list(entries)
        return _save_auth_store(auth_store)


def suppress_credential_source(provider_id: str, source: str) -> None:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.setdefault("suppressed_sources", {})
        provider_list = suppressed.setdefault(provider_id, [])
        if source not in provider_list:
            provider_list.append(source)
        _save_auth_store(auth_store)


def is_source_suppressed(provider_id: str, source: str) -> bool:
    try:
        suppressed = _load_auth_store().get("suppressed_sources", {})
        return source in suppressed.get(provider_id, [])
    except Exception:
        return False


def unsuppress_credential_source(provider_id: str, source: str) -> bool:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources")
        if not isinstance(suppressed, dict):
            return False
        provider_list = suppressed.get(provider_id)
        if not isinstance(provider_list, list) or source not in provider_list:
            return False
        provider_list.remove(source)
        if not provider_list:
            suppressed.pop(provider_id, None)
        if not suppressed:
            auth_store.pop("suppressed_sources", None)
        _save_auth_store(auth_store)
        return True


def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    return _load_provider_state(_load_auth_store(), provider_id)


def get_active_provider() -> Optional[str]:
    active = _load_auth_store().get("active_provider")
    return active if active in PROVIDER_REGISTRY else None


def is_known_auth_provider(provider_id: str) -> bool:
    return (provider_id or "").strip().lower() in PROVIDER_REGISTRY


def get_auth_provider_display_name(provider_id: str) -> str:
    pconfig = PROVIDER_REGISTRY.get((provider_id or "").strip().lower())
    return pconfig.name if pconfig else provider_id


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False
        providers = auth_store.setdefault("providers", {})
        pool = auth_store.setdefault("credential_pool", {})
        cleared = False
        if isinstance(providers, dict) and target in providers:
            del providers[target]
            cleared = True
        if isinstance(pool, dict) and target in pool:
            del pool[target]
            cleared = True
        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
            cleared = True
        if cleared:
            _save_auth_store(auth_store)
        return cleared


def deactivate_provider() -> None:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = None
        _save_auth_store(auth_store)


def _decode_jwt_claims(token: Any) -> Dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


def _read_codex_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    if _lock:
        with _auth_store_lock():
            auth_store = _load_auth_store()
    else:
        auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "openai-codex")
    if not state:
        raise AuthError(
            "No Codex credentials stored. Run `hermes auth add openai-codex` to authenticate.",
            provider="openai-codex",
            code="codex_auth_missing",
            relogin_required=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError(
            "Codex auth state is missing tokens. Run `hermes auth add openai-codex`.",
            provider="openai-codex",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthError("Codex auth is missing access_token.", provider="openai-codex", relogin_required=True)
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError("Codex auth is missing refresh_token.", provider="openai-codex", relogin_required=True)
    return {"tokens": tokens, "last_refresh": state.get("last_refresh")}


def _save_codex_tokens(tokens: Dict[str, str], last_refresh: str = None) -> None:
    if last_refresh is None:
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "openai-codex") or {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = "chatgpt"
        _save_provider_state(auth_store, "openai-codex", state)
        _save_auth_store(auth_store)


def refresh_codex_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    del access_token
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError("Codex auth is missing refresh_token.", provider="openai-codex", relogin_required=True)
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )
    if response.status_code != 200:
        relogin = response.status_code in (400, 401, 403)
        message = f"Codex token refresh failed with status {response.status_code}."
        try:
            payload = response.json()
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                message = f"Codex token refresh failed: {error['message']}"
            elif isinstance(error, str):
                message = f"Codex token refresh failed: {payload.get('error_description') or error}"
        except Exception:
            pass
        raise AuthError(message, provider="openai-codex", code="codex_refresh_failed", relogin_required=relogin)
    payload = response.json()
    refreshed_access = payload.get("access_token")
    if not isinstance(refreshed_access, str) or not refreshed_access.strip():
        raise AuthError("Codex token refresh response was missing access_token.", provider="openai-codex", relogin_required=True)
    updated = {
        "access_token": refreshed_access.strip(),
        "refresh_token": refresh_token.strip(),
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    next_refresh = payload.get("refresh_token")
    if isinstance(next_refresh, str) and next_refresh.strip():
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _refresh_codex_auth_tokens(tokens: Dict[str, str], timeout_seconds: float) -> Dict[str, str]:
    refreshed = refresh_codex_oauth_pure(
        str(tokens.get("access_token", "") or ""),
        str(tokens.get("refresh_token", "") or ""),
        timeout_seconds=timeout_seconds,
    )
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]
    _save_codex_tokens(updated_tokens)
    return updated_tokens


def _import_codex_cli_tokens() -> Optional[Dict[str, str]]:
    codex_home = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        payload = json.loads(auth_path.read_text())
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token or not refresh_token:
            return None
        if _codex_access_token_is_expiring(access_token, 0):
            return None
        return dict(tokens)
    except Exception:
        return None


def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    data = _read_codex_tokens()
    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = float(os.getenv("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", "20"))
    should_refresh = bool(force_refresh)
    if not should_refresh and refresh_if_expiring:
        should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)
    if should_refresh:
        with _auth_store_lock(timeout_seconds=max(AUTH_LOCK_TIMEOUT_SECONDS, refresh_timeout_seconds + 5.0)):
            data = _read_codex_tokens(_lock=False)
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()
            should_refresh = bool(force_refresh)
            if not should_refresh and refresh_if_expiring:
                should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)
            if should_refresh:
                tokens = _refresh_codex_auth_tokens(tokens, refresh_timeout_seconds)
                access_token = str(tokens.get("access_token", "") or "").strip()
    base_url = os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL
    return {
        "provider": "openai-codex",
        "base_url": base_url,
        "api_key": access_token,
        "source": "codex-auth",
        "last_refresh": data.get("last_refresh"),
        "auth_mode": "chatgpt",
    }


def get_codex_auth_status() -> Dict[str, Any]:
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("openai-codex")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
                if api_key and not _codex_access_token_is_expiring(api_key, 0):
                    return {
                        "logged_in": True,
                        "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        "auth_mode": "chatgpt",
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "api_key": api_key,
                    }
    except Exception:
        pass
    try:
        creds = resolve_codex_runtime_credentials()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
        }
    except AuthError as exc:
        return {"logged_in": False, "auth_store": str(_auth_file_path()), "error": str(exc)}


def resolve_provider(
    requested: Optional[str] = None,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> str:
    normalized = (requested or "auto").strip().lower()
    aliases = {
        "codex": "openai-codex",
        "chatgpt": "openai-codex",
        "gpt": "openai",
        "openai-api": "openai",
        "or": "openrouter",
        "open-router": "openrouter",
        "aigateway": "ai-gateway",
        "vercel": "ai-gateway",
        "vercel-ai-gateway": "ai-gateway",
        "custom": "custom",
        "local": "custom",
        "ollama": "custom",
        "vllm": "custom",
        "llamacpp": "custom",
        "llama.cpp": "custom",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"openai", "openrouter", "ai-gateway", "openai-codex", "custom"}:
        return normalized
    if normalized != "auto":
        raise AuthError(
            f"Unknown provider '{normalized}'. Hermes Simple supports OpenAI GPT, Codex, GPT API aggregators, and custom OpenAI-compatible endpoints.",
            provider=normalized,
            code="invalid_provider",
        )
    if explicit_base_url:
        return "custom"
    if explicit_api_key:
        return "openai"
    active = get_active_provider()
    if active == "openai-codex":
        status = get_codex_auth_status()
        if status.get("logged_in"):
            return "openai-codex"
    if has_usable_secret(get_env_value("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")):
        return "openai"
    if has_usable_secret(get_env_value("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")):
        return "openrouter"
    if has_usable_secret(get_env_value("AI_GATEWAY_API_KEY") or os.getenv("AI_GATEWAY_API_KEY", "")):
        return "ai-gateway"
    raise AuthError(
        "No inference provider configured. Set OPENAI_API_KEY or run `hermes auth add openai-codex`.",
        code="no_provider_configured",
    )


def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False, "logged_in": False}
    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)
    env_url = os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""
    return {
        "configured": bool(api_key),
        "logged_in": bool(api_key),
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source,
        "base_url": (env_url or pconfig.inference_base_url).rstrip("/"),
    }


def get_auth_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    target = (provider_id or get_active_provider() or "openai").strip().lower()
    if target == "openai-codex":
        return get_codex_auth_status()
    if target in {"openai", "openrouter", "ai-gateway"}:
        return get_api_key_provider_status(target)
    return {"logged_in": False, "configured": False, "provider": target, "error": "provider removed in Hermes Simple"}


def _resolve_api_key_provider_secret(provider_id: str, pconfig: ProviderConfig) -> tuple[str, str]:
    for env_var in pconfig.api_key_env_vars:
        val = (get_env_value(env_var) or os.getenv(env_var, "") or "").strip()
        if has_usable_secret(val):
            return val, env_var
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.peek()
            if entry:
                key = str(getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "")).strip()
                if has_usable_secret(key):
                    return key, f"credential_pool:{provider_id}"
    except Exception:
        pass
    return "", ""


def resolve_api_key_provider_credentials(provider_id: str) -> Dict[str, Any]:
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(f"Provider '{provider_id}' is not supported in Hermes Simple.", provider=provider_id, code="invalid_provider")
    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)
    env_url = os.getenv(pconfig.base_url_env_var, "").strip().rstrip("/") if pconfig.base_url_env_var else ""
    return {
        "provider": provider_id,
        "api_key": api_key,
        "base_url": env_url or pconfig.inference_base_url,
        "source": key_source or "default",
    }


def resolve_external_process_provider_credentials(provider_id: str) -> Dict[str, Any]:
    raise AuthError(f"Provider '{provider_id}' was removed in Hermes Simple.", provider=provider_id, code="provider_removed")


def get_external_process_provider_status(provider_id: str) -> Dict[str, Any]:
    return {"configured": False, "logged_in": False, "provider": provider_id, "error": "provider removed in Hermes Simple"}


def _update_config_for_provider(provider_id: str, inference_base_url: str, default_model: Optional[str] = None) -> Path:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = provider_id if provider_id in PROVIDER_REGISTRY else None
        _save_auth_store(auth_store)
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = read_raw_config()
    current_model = config.get("model")
    if isinstance(current_model, dict):
        model_cfg = dict(current_model)
    elif isinstance(current_model, str) and current_model.strip():
        model_cfg = {"default": current_model.strip()}
    else:
        model_cfg = {}
    model_cfg["provider"] = provider_id
    if inference_base_url:
        model_cfg["base_url"] = inference_base_url.rstrip("/")
    else:
        model_cfg.pop("base_url", None)
    model_cfg.pop("api_key", None)
    model_cfg.pop("api_mode", None)
    if default_model and not model_cfg.get("default"):
        model_cfg["default"] = default_model
    config["model"] = model_cfg
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _get_config_provider() -> Optional[str]:
    try:
        model = read_raw_config().get("model")
    except Exception:
        return None
    if not isinstance(model, dict):
        return None
    provider = model.get("provider")
    return provider.strip().lower() if isinstance(provider, str) and provider.strip() else None


def _config_provider_matches(provider_id: Optional[str]) -> bool:
    return bool(provider_id) and _get_config_provider() == provider_id.strip().lower()


def _logout_default_provider_from_config() -> Optional[str]:
    provider = _get_config_provider()
    return provider if provider in PROVIDER_REGISTRY else None


def _reset_config_provider() -> Path:
    config_path = get_config_path()
    if not config_path.exists():
        return config_path
    config = read_raw_config()
    model = config.get("model")
    if isinstance(model, dict):
        model["provider"] = "openai"
        model["base_url"] = DEFAULT_OPENAI_BASE_URL
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _prompt_model_selection(
    model_ids: List[str],
    current_model: str = "",
    pricing: Optional[Dict[str, Dict[str, str]]] = None,
    unavailable_models: Optional[List[str]] = None,
    portal_url: str = "",
) -> Optional[str]:
    del pricing, unavailable_models, portal_url
    ordered = []
    if current_model and current_model in model_ids:
        ordered.append(current_model)
    for mid in model_ids:
        if mid not in ordered:
            ordered.append(mid)
    print("Select default model:")
    for idx, mid in enumerate(ordered, 1):
        marker = " (current)" if mid == current_model else ""
        print(f"  {idx}. {mid}{marker}")
    print(f"  {len(ordered) + 1}. Enter custom model name")
    print(f"  {len(ordered) + 2}. Skip")
    try:
        choice = input(f"Choice [1-{len(ordered) + 2}] (default: skip): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not choice:
        return None
    try:
        idx = int(choice)
    except ValueError:
        return None
    if 1 <= idx <= len(ordered):
        return ordered[idx - 1]
    if idx == len(ordered) + 1:
        custom = input("Enter model name: ").strip()
        return custom or None
    return None


def _save_model_choice(model_id: str) -> None:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    if isinstance(config.get("model"), dict):
        config["model"]["default"] = model_id
    else:
        config["model"] = {"default": model_id}
    save_config(config)


def login_command(args) -> None:
    del args
    print("Use `hermes auth add openai-codex` for Codex OAuth, or set OPENAI_API_KEY for GPT models.")
    raise SystemExit(0)


def _login_openai_codex(args, pconfig: ProviderConfig, *, force_new_login: bool = False) -> None:
    del args, pconfig
    if not force_new_login:
        try:
            existing = resolve_codex_runtime_credentials()
            key = existing.get("api_key", "")
            if isinstance(key, str) and key and not _codex_access_token_is_expiring(key, 60):
                config_path = _update_config_for_provider("openai-codex", existing.get("base_url", DEFAULT_CODEX_BASE_URL))
                print("Existing Codex credentials found.")
                print(f"  Config updated: {config_path} (model.provider=openai-codex)")
                return
        except AuthError:
            pass
        cli_tokens = _import_codex_cli_tokens()
        if cli_tokens:
            _save_codex_tokens(cli_tokens)
            config_path = _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)
            print("Imported Codex CLI credentials.")
            print(f"  Config updated: {config_path} (model.provider=openai-codex)")
            return
    print()
    print("Signing in to OpenAI Codex...")
    creds = _codex_device_code_login()
    _save_codex_tokens(creds["tokens"], creds.get("last_refresh"))
    config_path = _update_config_for_provider("openai-codex", creds.get("base_url", DEFAULT_CODEX_BASE_URL))
    print("Login successful!")
    print(f"  Config updated: {config_path} (model.provider=openai-codex)")


def _codex_device_code_login() -> Dict[str, Any]:
    issuer = "https://auth.openai.com"
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                f"{issuer}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
    except Exception as exc:
        raise AuthError(f"Failed to request device code: {exc}", provider="openai-codex", code="device_code_request_failed")
    if resp.status_code != 200:
        raise AuthError(f"Device code request returned status {resp.status_code}.", provider="openai-codex", code="device_code_request_error")
    device_data = resp.json()
    user_code = device_data.get("user_code", "")
    device_auth_id = device_data.get("device_auth_id", "")
    poll_interval = max(3, int(device_data.get("interval", "5")))
    if not user_code or not device_auth_id:
        raise AuthError("Device code response missing required fields.", provider="openai-codex", code="device_code_incomplete")
    print("To continue, follow these steps:\n")
    print("  1. Open this URL in your browser:")
    print(f"     {issuer}/codex/device\n")
    print("  2. Enter this code:")
    print(f"     {user_code}\n")
    print("Waiting for sign-in... (press Ctrl+C to cancel)")
    max_wait = 15 * 60
    start = time.monotonic()
    code_resp = None
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while time.monotonic() - start < max_wait:
                time.sleep(poll_interval)
                poll_resp = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )
                if poll_resp.status_code == 200:
                    code_resp = poll_resp.json()
                    break
                if poll_resp.status_code in (403, 404):
                    continue
                raise AuthError(f"Device auth polling returned status {poll_resp.status_code}.", provider="openai-codex")
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    if code_resp is None:
        raise AuthError("Login timed out after 15 minutes.", provider="openai-codex", code="device_code_timeout")
    authorization_code = code_resp.get("authorization_code", "")
    code_verifier = code_resp.get("code_verifier", "")
    if not authorization_code or not code_verifier:
        raise AuthError("Device auth response missing authorization data.", provider="openai-codex")
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        token_resp = client.post(
            CODEX_OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": f"{issuer}/deviceauth/callback",
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_resp.status_code != 200:
        raise AuthError(f"Token exchange returned status {token_resp.status_code}.", provider="openai-codex")
    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")
    if not access_token:
        raise AuthError("Token exchange did not return an access_token.", provider="openai-codex")
    return {
        "tokens": {"access_token": access_token, "refresh_token": refresh_token},
        "base_url": os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL,
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "auth_mode": "chatgpt",
        "source": "device-code",
    }


def logout_command(args) -> None:
    provider_id = getattr(args, "provider", None)
    if provider_id and not is_known_auth_provider(provider_id):
        print(f"Unknown provider: {provider_id}")
        raise SystemExit(1)
    active = get_active_provider()
    target = provider_id or active or _logout_default_provider_from_config()
    if not target:
        print("No provider is currently logged in.")
        return
    config_matches = _config_provider_matches(target)
    provider_name = get_auth_provider_display_name(target)
    if clear_provider_auth(target) or config_matches:
        _reset_config_provider()
        print(f"Logged out of {provider_name}.")
        print("Set OPENAI_API_KEY or run `hermes auth add openai-codex` to use Hermes.")
    else:
        print(f"No auth state found for {provider_name}.")


def is_provider_explicitly_configured(provider_id: str) -> bool:
    return (provider_id or "").strip().lower() in PROVIDER_REGISTRY and _config_provider_matches(provider_id)
