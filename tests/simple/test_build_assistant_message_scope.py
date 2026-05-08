import inspect
from types import SimpleNamespace

from run_agent import AIAgent


def test_build_assistant_message_scope_is_gpt_codex_only():
    source = inspect.getsource(AIAgent._build_assistant_message)
    forbidden_fragments = [
        "reasoning_details",
        "extra_content",
        "_needs_deepseek_tool_reasoning",
        "DeepSeek",
        "Kimi",
        "Gemini",
        "Moonshot",
        "OpenRouter",
        "Anthropic",
    ]
    offenders = [fragment for fragment in forbidden_fragments if fragment in source]

    assert offenders == []


def test_build_assistant_message_preserves_codex_state_and_tool_ids():
    agent = AIAgent.__new__(AIAgent)
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None

    tool_call = SimpleNamespace(
        id="call_abc|fc_abc",
        call_id="call_abc",
        response_item_id="fc_abc",
        type="function",
        function=SimpleNamespace(name="terminal", arguments='{"cmd":"pwd"}'),
    )
    assistant_message = SimpleNamespace(
        content="visible",
        reasoning="reasoning",
        reasoning_content=None,
        tool_calls=[tool_call],
        codex_reasoning_items=[{"id": "rs_1"}],
        codex_message_items=[{"id": "msg_1"}],
    )

    msg = agent._build_assistant_message(assistant_message, "tool_calls")

    assert msg["role"] == "assistant"
    assert msg["content"] == "visible"
    assert msg["reasoning"] == "reasoning"
    assert msg["finish_reason"] == "tool_calls"
    assert msg["codex_reasoning_items"] == [{"id": "rs_1"}]
    assert msg["codex_message_items"] == [{"id": "msg_1"}]
    assert msg["tool_calls"] == [
        {
            "id": "call_abc",
            "call_id": "call_abc",
            "response_item_id": "fc_abc",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
        }
    ]


def test_api_replay_strips_provider_reasoning_fields():
    agent = AIAgent.__new__(AIAgent)
    source_msg = {
        "role": "assistant",
        "content": "visible",
        "reasoning": "internal only",
        "reasoning_content": "provider field",
        "reasoning_details": [{"summary": "provider detail"}],
    }
    api_msg = dict(source_msg)

    agent._copy_reasoning_content_for_api(source_msg, api_msg)

    assert "reasoning" not in api_msg
    assert "reasoning_content" not in api_msg
    assert "reasoning_details" not in api_msg
