import inspect

from run_agent import AIAgent, _build_chat_api_kwargs, _build_codex_api_kwargs


def test_chat_completions_kwargs_do_not_inject_provider_extra_body():
    kwargs = _build_chat_api_kwargs(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "high"},
        supports_reasoning=True,
        is_kimi=True,
        is_qwen_portal=True,
        is_nvidia_nim=True,
        is_github_models=True,
        is_lmstudio=True,
        is_nous=True,
        ollama_num_ctx=32768,
        extra_body_additions={"provider_flag": True},
    )

    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_chat_kwargs_builder_has_no_provider_reasoning_branches():
    source = inspect.getsource(_build_chat_api_kwargs)
    forbidden_fragments = [
        "extra_body",
        "reasoning_effort",
        "supports_reasoning",
        "github_reasoning_extra",
        "ollama_num_ctx",
        "extra_body_additions",
        "vl_high_resolution_images",
        "is_github_models",
        "is_lmstudio",
        "is_nous",
        "is_qwen_portal",
        "is_nvidia_nim",
        "is_kimi",
        "qwen_prepare_fn",
        "qwen_prepare_inplace_fn",
        "qwen_session_metadata",
    ]

    offenders = [fragment for fragment in forbidden_fragments if fragment in source]
    assert offenders == []


def test_agent_has_no_provider_reasoning_extra_body_helpers():
    source = inspect.getsource(AIAgent)
    forbidden_fragments = [
        "_supports_reasoning_extra_body",
        "_github_models_reasoning_extra_body",
        "_lmstudio_reasoning_options_cached",
        "_resolve_lmstudio_summary_reasoning_effort",
        "summary_extra_body",
        "github_reasoning_extra",
        "lmstudio_reasoning_options",
        "_is_qwen",
        "_is_nvidia",
        "_is_kimi",
        "_qwen_meta",
    ]

    offenders = [fragment for fragment in forbidden_fragments if fragment in source]
    assert offenders == []


def test_codex_kwargs_keep_openai_reasoning_shape_without_provider_branches():
    kwargs = _build_codex_api_kwargs(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "high"},
        session_id="session-123",
        is_github_responses=True,
        is_xai_responses=True,
    )

    assert kwargs["prompt_cache_key"] == "session-123"
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kwargs["include"] == ["reasoning.encrypted_content"]
