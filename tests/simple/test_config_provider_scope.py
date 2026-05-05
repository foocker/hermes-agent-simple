from hermes_cli import config as cfg


def test_config_provider_env_surface_is_simple_only():
    provider_env = {
        name
        for name, info in cfg.OPTIONAL_ENV_VARS.items()
        if info.get("category") == "provider"
    }
    assert provider_env == {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "AI_GATEWAY_API_KEY",
        "AI_GATEWAY_BASE_URL",
    }


def test_config_user_facing_text_uses_simple_providers():
    forbidden = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "AWS_REGION",
        "AWS_PROFILE",
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_FOUNDRY_BASE_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
        "Nous Portal",
        "Bedrock",
        "Anthropic OAuth",
        "Z.AI",
        "Kimi / Moonshot",
        "MiniMax",
        "anthropic/claude-sonnet",
    ]
    surfaces = [
        cfg._FALLBACK_COMMENT,
        cfg._COMMENTED_SECTIONS,
        "\n".join(
            str(value)
            for info in cfg.OPTIONAL_ENV_VARS.values()
            for value in info.values()
        ),
    ]
    text = "\n".join(surfaces)
    offenders = [fragment for fragment in forbidden if fragment in text]
    assert offenders == []
