from pathlib import Path


def test_agent_runtime_has_no_anthropic_prompt_cache_marker_injection():
    repo_root = Path(__file__).resolve().parents[2]
    checked_files = [repo_root / "run_agent.py", *sorted((repo_root / "agent").glob("*.py"))]
    forbidden_fragments = [
        "agent.prompt_caching",
        "apply_anthropic_cache_control",
        "_anthropic_prompt_cache_policy",
        "_use_prompt_caching",
        "_use_native_cache_layout",
        "_cache_ttl",
        "cache_control",
        "cache_ttl",
        "native_anthropic",
    ]

    offenders = []
    for path in checked_files:
        text = path.read_text()
        for fragment in forbidden_fragments:
            if fragment in text:
                offenders.append(f"{path.relative_to(repo_root)} contains {fragment}")

    assert offenders == []
