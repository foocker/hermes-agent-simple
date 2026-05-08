from pathlib import Path


def test_runtime_has_no_openrouter_nous_identity_or_provider_prefs():
    repo_root = Path(__file__).resolve().parents[2]
    checked_files = [
        repo_root / "run_agent.py",
        repo_root / "agent" / "auxiliary_client.py",
        repo_root / "cli.py",
        repo_root / "gateway" / "run.py",
        repo_root / "batch_runner.py",
        repo_root / "tools" / "delegate_tool.py",
        repo_root / "cron" / "scheduler.py",
        repo_root / "tui_gateway" / "server.py",
        repo_root / "hermes_cli" / "tips.py",
        repo_root / "cli-config.yaml.example",
        repo_root / "agent" / "prompt_builder.py",
        repo_root / "hermes_cli" / "model_catalog.py",
        repo_root / "hermes_cli" / "config.py",
        repo_root / "tools" / "skills_hub.py",
        repo_root / "hermes_cli" / "setup.py",
        repo_root / "hermes_cli" / "tools_config.py",
        repo_root / "hermes_cli" / "fallback_cmd.py",
        repo_root / "hermes_cli" / "main.py",
        repo_root / "trajectory_compressor.py",
        repo_root / "agent" / "model_metadata.py",
    ]
    forbidden_fragments = [
        "nousresearch.com",
        "portal.nousresearch.com",
        "X-OpenRouter-Title",
        "X-OpenRouter-Categories",
        "_openrouter_prewarm_done",
        "providers_allowed",
        "providers_ignored",
        "providers_order",
        "provider_sort",
        "provider_require_parameters",
        "provider_data_collection",
        "provider_preferences",
        "require_parameters",
        "data_collection",
    ]

    offenders = []
    for path in checked_files:
        text = path.read_text()
        for fragment in forbidden_fragments:
            if fragment in text:
                offenders.append(f"{path.relative_to(repo_root)} contains {fragment}")

    assert offenders == []
