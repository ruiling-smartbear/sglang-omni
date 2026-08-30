# Real-template equivalence check for the bounded generation-prompt delta
# (verl #7617). Run on CPU with network access; compares the bounded pseudo
# render against the upstream full-history render on real Qwen tokenizers.
import sys

sys.path.insert(0, "/tmp/verl")

from typing import Any

from transformers import AutoTokenizer

from verl.utils.tokenizer.continuous_token import (
    _SYNTHETIC_SYSTEM_MESSAGE,
    _SYNTHETIC_USER_MESSAGE,
    QwenContinuousTokenBuilder,
)

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B"]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]


class BoundedQwenBuilder(QwenContinuousTokenBuilder):
    """Mirror of the PR's base-class _tokenize_generation_prompt_delta."""

    def _tokenize_generation_prompt_delta(self, updated_messages, *, tools=None):
        if not updated_messages:
            return self.render_delta_token_id(
                [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE], [], add_generation_prompt=True, tools=tools
            )
        last_message = updated_messages[-1]
        if last_message.get("role") == "tool":
            pseudo_prefix = [
                _SYNTHETIC_SYSTEM_MESSAGE,
                _SYNTHETIC_USER_MESSAGE,
                self._synthetic_assistant_for_tools([last_message]),
                last_message,
            ]
        else:
            pseudo_prefix = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, last_message]
        return self.render_delta_token_id(pseudo_prefix, [], add_generation_prompt=True, tools=tools)


def scenarios() -> list[tuple[str, list[dict[str, Any]]]]:
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": {}}}
    base = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Find the population of Pittsburgh."},
    ]
    asst = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    tool = {"role": "tool", "content": "302,971", "tool_call_id": "call-1", "name": "lookup"}
    long_tail = []
    for turn in range(12):
        long_tail.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{turn}", "type": "function", "function": {"name": "lookup", "arguments": {}}}],
            }
        )
        long_tail.append({"role": "tool", "content": f"partial {turn}", "tool_call_id": f"c{turn}"})
    return [
        ("user-last", base),
        ("tool-last", base + [asst, tool]),
        ("multi-tool-last", base + [{**asst, "tool_calls": [tool_call, {"id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": {}}}]}, tool, {**tool, "tool_call_id": "call-2"}]),
        ("assistant-then-user", base + [{"role": "assistant", "content": "Let me check.", "reasoning_content": "thinking..."}, {"role": "user", "content": "Please do."}]),
        ("long-tool-loop", base + long_tail),
    ]


def main() -> int:
    failures = 0
    for model in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {model}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        for kwargs in ({}, {"enable_thinking": False}):
            legacy = QwenContinuousTokenBuilder(tok, chat_template_kwargs=dict(kwargs))
            bounded = BoundedQwenBuilder(tok, chat_template_kwargs=dict(kwargs))
            for name, conversation in scenarios():
                for tools in (None, TOOLS):
                    try:
                        want = legacy._tokenize_generation_prompt_delta(conversation, tools=tools)
                        got = bounded._tokenize_generation_prompt_delta(conversation, tools=tools)
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        print(f"[ERROR] {model} kwargs={kwargs} {name} tools={bool(tools)}: {type(exc).__name__}: {str(exc)[:110]}")
                        continue
                    status = "ok" if got == want else "MISMATCH"
                    if got != want:
                        failures += 1
                        print(f"[{status}] {model} kwargs={kwargs} {name} tools={bool(tools)} want={want} got={got}")
                    else:
                        print(f"[{status}] {model} kwargs={kwargs} scenario={name} tools={bool(tools)} delta_len={len(got)}")
    print("TEMPLATE_CHECK_" + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
