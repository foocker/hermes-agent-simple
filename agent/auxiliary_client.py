"""GPT/Codex-only auxiliary client helpers.

This module intentionally keeps the public helper names used by compression,
vision, search, and title generation, while removing third-party provider
adapters. Supported runtime shapes are:

- OpenAI-compatible Chat Completions
- OpenAI Codex / Responses API through ``CodexAuxiliaryClient``
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import parse_qs, urlparse, urlunparse

from hermes_cli.config import get_hermes_home, load_config
from utils import base_url_hostname, normalize_proxy_env_vars

if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401

logger = logging.getLogger(__name__)


_OPENAI_CLS_CACHE: Optional[type] = None


def _load_openai_cls() -> type:
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls

        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())


OpenAI = _OpenAIProxy()

OMIT_TEMPERATURE: object = object()

_AUTH_JSON_PATH = get_hermes_home() / "auth.json"
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"

auxiliary_is_nous = False


def _extract_url_query_params(url: str):
    parsed = urlparse(url)
    if parsed.query:
        clean = urlunparse(parsed._replace(query=""))
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return clean, params
    return url, None


def _to_openai_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/anthropic"):
        return normalized[: -len("/anthropic")] + "/v1"
    if normalized.endswith("/coding"):
        return normalized + "/v1"
    return normalized


def _validate_proxy_env_urls() -> None:
    normalize_proxy_env_vars()


def _validate_base_url(base_url: str) -> None:
    if not str(base_url or "").strip():
        raise ValueError("base_url is required")


def _codex_cloudflare_headers(access_token: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
        "Origin": "https://chatgpt.com",
        "Referer": "https://chatgpt.com/",
    }
    account_id = ""
    try:
        from hermes_cli.auth import _read_codex_tokens

        token_data = _read_codex_tokens()
        tokens = token_data.get("tokens") or {}
        account_id = str(tokens.get("account_id", "") or "").strip()
    except Exception:
        account_id = ""
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def _read_codex_access_token() -> Optional[str]:
    try:
        from hermes_cli.auth import _read_codex_tokens

        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None
        try:
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                return None
        except Exception:
            pass
        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth: %s", exc)
        return None


def _convert_content_for_responses(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    converted = []
    for part in content:
        if isinstance(part, str):
            converted.append({"type": "input_text", "text": part})
            continue
        if not isinstance(part, dict):
            converted.append({"type": "input_text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text"}:
            converted.append({"type": "input_text", "text": str(part.get("text", "") or "")})
        elif ptype in {"image_url", "input_image"}:
            image = part.get("image_url")
            if isinstance(image, dict):
                image = image.get("url")
            converted.append({"type": "input_image", "image_url": str(image or "")})
        else:
            converted.append(part)
    return converted


class _CodexCompletionsAdapter:
    def __init__(self, real_client: Any, model: str):
        self._real_client = real_client
        self._model = model

    def _messages_to_input(self, messages: list) -> list:
        items = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role == "system":
                role = "developer"
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id") or msg.get("call_id") or "",
                        "output": msg.get("content") or "",
                    }
                )
                continue
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _convert_content_for_responses(msg.get("content")),
                }
            )
        return items

    def _tools_for_responses(self, tools: Optional[list]) -> Optional[list]:
        if not tools:
            return None
        converted = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            if not isinstance(fn, dict):
                continue
            converted.append(
                {
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return converted or None

    def create(self, **kwargs) -> Any:
        model = kwargs.get("model") or self._model
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "input": self._messages_to_input(kwargs.get("messages", [])),
        }
        tools = self._tools_for_responses(kwargs.get("tools"))
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
        max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if max_tokens is not None:
            create_kwargs["max_output_tokens"] = max_tokens
        if kwargs.get("timeout") is not None:
            create_kwargs["timeout"] = kwargs["timeout"]

        response = self._real_client.responses.create(**create_kwargs)
        return _normalize_codex_response(response, model)


def _normalize_codex_response(response: Any, model: str) -> Any:
    content_parts: List[str] = []
    tool_calls = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else "")
        if item_type == "message":
            for part in getattr(item, "content", None) or (item.get("content", []) if isinstance(item, dict) else []):
                text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                if text:
                    content_parts.append(str(text))
        elif item_type == "function_call":
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
            args = getattr(item, "arguments", None) or (item.get("arguments") if isinstance(item, dict) else "{}")
            call_id = getattr(item, "call_id", None) or (item.get("call_id") if isinstance(item, dict) else "")
            tool_calls.append(
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(name=name, arguments=args),
                )
            )
    output_text = getattr(response, "output_text", "") or ""
    if output_text and not content_parts:
        content_parts.append(str(output_text))
    message = SimpleNamespace(
        content="\n".join(content_parts),
        tool_calls=tool_calls or None,
        reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls" if tool_calls else "stop")],
        model=model,
        usage=getattr(response, "usage", None),
    )


class _CodexChatShim:
    def __init__(self, adapter: _CodexCompletionsAdapter):
        self.completions = adapter


class CodexAuxiliaryClient:
    def __init__(self, real_client: Any, model: str):
        self._real_client = real_client
        self.chat = _CodexChatShim(_CodexCompletionsAdapter(real_client, model))
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class _AsyncCodexCompletionsAdapter:
    def __init__(self, sync_adapter: _CodexCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCodexChatShim:
    def __init__(self, adapter: _AsyncCodexCompletionsAdapter):
        self.completions = adapter


class AsyncCodexAuxiliaryClient:
    def __init__(self, sync_wrapper: CodexAuxiliaryClient):
        self.chat = _AsyncCodexChatShim(_AsyncCodexCompletionsAdapter(sync_wrapper.chat.completions))
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url


def _read_main_provider() -> str:
    try:
        cfg = load_config()
        return str((cfg.get("model") or {}).get("provider") or "custom").strip().lower()
    except Exception:
        return "custom"


def _read_main_model() -> str:
    try:
        cfg = load_config()
        return str((cfg.get("model") or {}).get("model") or "").strip()
    except Exception:
        return ""


def _current_custom_base_url() -> str:
    return (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    ).strip().rstrip("/")


def _fixed_temperature_for_model(model: Optional[str], base_url: Optional[str] = None) -> "Optional[float] | object":
    return None


def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized in {"codex", "openai-codex"}:
        return "openai-codex"
    if normalized in {"openai", "gpt", "custom", "main", "auto"}:
        return normalized
    return "custom"


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    cfg_provider = cfg_model = cfg_base_url = cfg_api_key = cfg_api_mode = None
    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None
        cfg_api_mode = str(task_config.get("api_mode", "")).strip() or None

    resolved_model = model or cfg_model
    resolved_api_mode = cfg_api_mode
    if base_url:
        return "custom", resolved_model, base_url, api_key, resolved_api_mode
    if provider:
        return provider, resolved_model, base_url, api_key, resolved_api_mode
    if task:
        if cfg_base_url:
            return "custom", resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, resolved_model, None, None, resolved_api_mode
    return "auto", resolved_model, None, None, resolved_api_mode


def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    if not task:
        return {}
    try:
        cfg = load_config()
    except Exception:
        return {}
    aux = cfg.get("auxiliary", {}) if isinstance(cfg, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    return task_config if isinstance(task_config, dict) else {}


def _get_task_timeout(task: str, default: float = 30.0) -> float:
    raw = _get_auxiliary_task_config(task).get("timeout") if task else None
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _build_openai_client(api_key: str, base_url: str, *, timeout: Optional[float] = None) -> Any:
    clean_base, default_query = _extract_url_query_params(_to_openai_base_url(base_url))
    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": clean_base}
    if default_query:
        kwargs["default_query"] = default_query
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def _build_codex_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    token = _read_codex_access_token()
    if not token:
        return None, None
    client = OpenAI(
        api_key=token,
        base_url=_CODEX_AUX_BASE_URL,
        default_headers=_codex_cloudflare_headers(token),
    )
    return CodexAuxiliaryClient(client, model), model


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    _validate_proxy_env_urls()
    provider = _normalize_aux_provider(provider)
    final_model = model or (main_runtime or {}).get("model") or _read_main_model() or "gpt-4o-mini"

    if provider == "auto":
        main_provider = _normalize_aux_provider(_read_main_provider())
        provider = "openai-codex" if main_provider == "openai-codex" else "custom"

    if api_mode == "codex_responses" or provider == "openai-codex":
        if raw_codex:
            token = _read_codex_access_token()
            if not token:
                return None, None
            return (
                OpenAI(
                    api_key=token,
                    base_url=_CODEX_AUX_BASE_URL,
                    default_headers=_codex_cloudflare_headers(token),
                ),
                final_model,
            )
        client, default_model = _build_codex_client(final_model)
        if client is None:
            return None, None
        return (_to_async_client(client, default_model, is_vision=is_vision) if async_mode else (client, default_model))

    base_url = explicit_base_url or (main_runtime or {}).get("base_url") or _current_custom_base_url()
    api_key = (
        explicit_api_key
        or (main_runtime or {}).get("api_key")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_TOKEN")
        or "no-key-required"
    )
    client = _build_openai_client(str(api_key), str(base_url), timeout=_get_task_timeout(""))
    return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode else (client, final_model))


def _to_async_client(sync_client: Any, model: str, is_vision: bool = False):
    from openai import AsyncOpenAI

    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model
    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    return AsyncOpenAI(**async_kwargs), model


def get_text_auxiliary_client(
    task: str = "",
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


def get_async_text_auxiliary_client(task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None):
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    client, final_model = resolve_provider_client(
        requested,
        model=resolved_model,
        async_mode=async_mode,
        explicit_base_url=resolved_base_url,
        explicit_api_key=resolved_api_key,
        api_mode=resolved_api_mode,
        is_vision=True,
    )
    return _normalize_aux_provider(requested), client, final_model


def get_available_vision_backends() -> List[str]:
    provider, client, _model = resolve_vision_provider_client()
    return [provider] if client is not None and provider else []


def get_auxiliary_extra_body() -> dict:
    return {}


def auxiliary_max_tokens_param(value: int) -> dict:
    return {"max_completion_tokens": value}


def _build_call_kwargs(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
    base_url: Optional[str] = None,
) -> dict:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        if provider == "openai-codex" or base_url_hostname(base_url or "") == "api.openai.com":
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _validate_llm_response(response: Any, task: str = None) -> Any:
    if not getattr(response, "choices", None):
        raise RuntimeError(f"Auxiliary LLM returned no choices for {task or 'task'}")
    return response


def call_llm(
    messages: list,
    provider: str = "auto",
    model: str = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    task: str = None,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    timeout: Optional[float] = None,
    extra_body: Optional[dict] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
):
    client, final_model = resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )
    if client is None or not final_model:
        raise RuntimeError("No GPT/Codex auxiliary client is configured")
    call_kwargs = _build_call_kwargs(
        _normalize_aux_provider(provider),
        final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=timeout or _get_task_timeout(task or "", 30.0),
        extra_body=extra_body,
        base_url=str(getattr(client, "base_url", "") or base_url or ""),
    )
    return _validate_llm_response(client.chat.completions.create(**call_kwargs), task)


async def async_call_llm(
    messages: list,
    provider: str = "auto",
    model: str = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    task: str = None,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    timeout: Optional[float] = None,
    extra_body: Optional[dict] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
):
    client, final_model = resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )
    if client is None or not final_model:
        raise RuntimeError("No GPT/Codex auxiliary client is configured")
    call_kwargs = _build_call_kwargs(
        _normalize_aux_provider(provider),
        final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=timeout or _get_task_timeout(task or "", 30.0),
        extra_body=extra_body,
        base_url=str(getattr(client, "base_url", "") or base_url or ""),
    )
    return _validate_llm_response(await client.chat.completions.create(**call_kwargs), task)


def extract_content_or_reasoning(response) -> str:
    try:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        if content:
            return str(content)
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            return str(reasoning)
    except Exception:
        pass
    return ""


def neuter_async_httpx_del() -> None:
    return None


def cleanup_stale_async_clients() -> None:
    return None


def shutdown_cached_clients() -> None:
    return None
