import importlib.util
from pathlib import Path


def test_agent_keeps_only_gpt_and_codex_provider_surface():
    repo_root = Path(__file__).resolve().parents[2]
    removed_modules = [
        "agent.transports",
        "agent.anthropic_adapter",
        "agent.bedrock_adapter",
        "agent.gemini_native_adapter",
        "agent.gemini_cloudcode_adapter",
        "agent.gemini_schema",
        "agent.google_code_assist",
        "agent.google_oauth",
        "agent.copilot_acp_client",
        "agent.lmstudio_reasoning",
        "agent.moonshot_schema",
        "agent.nous_rate_guard",
    ]

    for module_name in removed_modules:
        module_path = repo_root / Path(*module_name.split("."))
        assert not module_path.with_suffix(".py").exists()
        assert not module_path.is_dir()

    kept_modules = [
        "agent.image_gen_provider",
        "agent.image_gen_registry",
        "agent.image_routing",
        "agent.codex_responses_adapter",
        "run_agent",
    ]

    for module_name in kept_modules:
        assert importlib.util.find_spec(module_name) is not None


def test_agent_runtime_no_longer_references_removed_provider_adapters():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden_fragments = [
        "agent.anthropic_adapter",
        "agent.bedrock_adapter",
        "agent.gemini_native_adapter",
        "agent.gemini_cloudcode_adapter",
        "agent.gemini_schema",
        "agent.google_code_assist",
        "agent.google_oauth",
        "agent.copilot_acp_client",
        "agent.lmstudio_reasoning",
        "agent.moonshot_schema",
        "agent.nous_rate_guard",
        "agent.transports",
        "anthropic_messages",
        "bedrock_converse",
    ]
    checked_files = [repo_root / "run_agent.py", *sorted((repo_root / "agent").glob("*.py"))]

    offenders = []
    for path in checked_files:
        text = path.read_text()
        for fragment in forbidden_fragments:
            if fragment in text:
                offenders.append(f"{path.relative_to(repo_root)} contains {fragment}")

    assert offenders == []
