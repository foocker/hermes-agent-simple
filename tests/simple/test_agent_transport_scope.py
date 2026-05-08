from pathlib import Path


def test_agent_runtime_has_no_transport_abstraction_layer():
    repo_root = Path(__file__).resolve().parents[2]
    run_agent = repo_root / "run_agent.py"
    text = run_agent.read_text()
    forbidden_fragments = [
        "_get_transport(",
        "_SimpleChatTransport",
        "_SimpleCodexTransport",
        "ChatCompletionsTransport",
        "CodexResponsesTransport",
        "agent.transports",
    ]

    offenders = [fragment for fragment in forbidden_fragments if fragment in text]
    assert offenders == []
