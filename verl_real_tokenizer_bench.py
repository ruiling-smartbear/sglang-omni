# Real-tokenizer benchmark for verl #7617: legacy full-history generation-prompt
# delta vs the bounded render, driven through the public incremental API with
# real Qwen tokenizers and chat templates. CPU only.
import sys
import time

sys.path.insert(0, "/tmp/verl")

from transformers import AutoTokenizer

from verl.utils.tokenizer.continuous_token import (
    _SYNTHETIC_SYSTEM_MESSAGE,
    _SYNTHETIC_USER_MESSAGE,
    QwenContinuousTokenBuilder,
)

MODELS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"]
TOOL_RESULT = ("The lookup returned 302,971 residents as of the latest census estimate. " * 12)[:800]


class BoundedQwenBuilder(QwenContinuousTokenBuilder):
    """Mirror of PR #7619's base-class _tokenize_generation_prompt_delta."""

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


def run(builder, turns):
    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
        {"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."},
    ]
    total = 0.0
    slowest_turn = 0.0
    collected = []
    for turn in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call-{turn}", "type": "function", "function": {"name": "lookup", "arguments": {"q": str(turn)}}}
                ],
            }
        )
        previous = list(messages)
        messages.append({"role": "tool", "content": TOOL_RESULT, "tool_call_id": f"call-{turn}", "name": "lookup"})
        t0 = time.perf_counter()
        ids = builder.tokenize_non_assistant_incremental_messages(previous, messages)
        dt = time.perf_counter() - t0
        total += dt
        slowest_turn = max(slowest_turn, dt)
        collected.extend(ids)
    return total, slowest_turn, collected


def main():
    for model in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {model}: {type(exc).__name__}")
            continue
        for turns in (50, 100, 200):
            legacy_total, legacy_worst, legacy_ids = run(QwenContinuousTokenBuilder(tok), turns)
            bounded_total, bounded_worst, bounded_ids = run(BoundedQwenBuilder(tok), turns)
            same = legacy_ids == bounded_ids
            print(
                f"RESULT {model} turns={turns} "
                f"legacy_total={legacy_total:.2f}s (worst turn {legacy_worst*1000:.0f}ms) | "
                f"bounded_total={bounded_total:.3f}s (worst turn {bounded_worst*1000:.1f}ms) | "
                f"speedup={legacy_total / bounded_total:.0f}x | identical_tokens={same}"
            )
    print("REAL_TOKENIZER_BENCH_DONE")


if __name__ == "__main__":
    main()
