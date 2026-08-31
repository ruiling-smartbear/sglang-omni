# Rollout-level check of verl's DeepSeek continuous-token path on a real model.
#
# One tool turn, driven the way tool_agent_loop drives the builder:
#   build_initial_tokens -> model generates a tool call -> tool runs ->
#   merge_non_assistant_tokens (tokenize_non_assistant_incremental_messages,
#   the path #7630 fixes) -> model continues from the merged token ids.
# The model is served by an sglang server; the driver only needs transformers.
# Run once against verl before #7630 (the append raises) and once against
# current main (the model answers from the tool result).
import argparse
import json
import re
import sys
import time
import types
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--verl", required=True, help="path to a verl checkout")
parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
parser.add_argument("--server", default="http://127.0.0.1:30000")
parser.add_argument("--label", default="")
parser.add_argument("--max-new-tokens", type=int, default=900)
args = parser.parse_args()

for name, sub in (("verl", "verl"), ("verl.utils", "verl/utils")):
    stub = types.ModuleType(name)
    stub.__path__ = [f"{args.verl}/{sub}"]
    sys.modules[name] = stub
from transformers import AutoTokenizer  # noqa: E402
from verl.utils.tokenizer.continuous_token_wiring import get_continuous_token_builder_class  # noqa: E402

TOOL_NAME = "get_population"
TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the latest population estimate of a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
TOOL_ANSWER = "Pittsburgh: 302,971 residents (2023 census estimate)."
CALL_FORMAT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_population\n"
    "```json\n{\"city\": \"<city name>\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
)
SYSTEM_PROMPT = (
    "You can call one tool, get_population(city), which returns the latest population estimate of a city. "
    "You do not know any population figures yourself. To call the tool, end your reply with exactly this block "
    "and nothing after it:\n" + CALL_FORMAT + "\nAfter the tool result arrives, answer the user in one sentence."
)
USER_PROMPT = "What is the current population of Pittsburgh? Use the tool."
CALL_RE = re.compile(
    r"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\n```json\n(.*?)\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    re.S,
)


def generate(token_ids, max_new_tokens):
    payload = {
        "input_ids": token_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "skip_special_tokens": False},
    }
    request = urllib.request.Request(
        args.server + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    return body


def tail(text, n=260):
    return repr(text[-n:])


tag = f"[{args.label}] " if args.label else ""
tok = AutoTokenizer.from_pretrained(args.model)
builder = get_continuous_token_builder_class("deepseek")(tok)
eos_id = builder._eos_id
print(f"{tag}verl={args.verl} model={args.model} builder={type(builder).__name__} eos_id={eos_id}")

messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]
prompt_ids = builder.build_initial_tokens(messages, tools=TOOLS)
print(f"{tag}prompt: {len(prompt_ids)} tokens, tail {tail(tok.decode(prompt_ids), 120)}")

t0 = time.time()
first = generate(prompt_ids, args.max_new_tokens)
first_text = first["text"]
first_ids = first.get("output_ids") or tok.encode(first_text, add_special_tokens=False)
print(f"{tag}turn 1: {len(first_ids)} tokens in {time.time() - t0:.1f}s, finish={first['meta_info'].get('finish_reason')}")
print(f"{tag}turn 1 text tail: {tail(first_text, 400)}")

match = CALL_RE.search(first_text)
forced = False
if match and match.group(1) == TOOL_NAME:
    try:
        arguments = json.loads(match.group(2))
    except json.JSONDecodeError:
        arguments = {"city": "Pittsburgh"}
    assistant_ids = list(first_ids)
    assistant_content = first_text[: match.start()]
    print(f"{tag}model called the tool itself: {match.group(0)[-120:]!r}")
else:
    # The model did not produce the call format; stand in for it with a hand-written
    # assistant turn so the tool-append path is still exercised.
    forced = True
    arguments = {"city": "Pittsburgh"}
    assistant_content = "<think>\nI need the tool for this.\n</think>\n\n"
    assistant_text = assistant_content + CALL_FORMAT.replace("<city name>", "Pittsburgh")
    assistant_ids = tok.encode(assistant_text, add_special_tokens=False)
    print(f"{tag}model did not call the tool; using a hand-written tool-call turn")
if assistant_ids[-1] != eos_id:
    assistant_ids.append(eos_id)

assistant_message = {
    "role": "assistant",
    "content": assistant_content,
    "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": TOOL_NAME, "arguments": arguments}}],
}
tool_message = {"role": "tool", "content": TOOL_ANSWER, "tool_call_id": "call_0"}

runtime_ids = builder.merge_assistant_tokens(prompt_ids, assistant_ids).token_ids
previous = messages + [assistant_message]
updated = previous + [tool_message]
try:
    merged = builder.merge_non_assistant_tokens(previous, updated, runtime_ids, tools=TOOLS)
except Exception as exc:  # noqa: BLE001
    print(f"{tag}TOOL APPEND FAILED: {type(exc).__name__}: {str(exc)[:160]}")
    print(f"{tag}RESULT: append_failed forced_call={forced}")
    raise SystemExit(0)
appended = merged.token_ids[len(runtime_ids):]
print(f"{tag}tool append: {len(appended)} tokens -> {tok.decode(appended)!r}")

t0 = time.time()
second = generate(merged.token_ids, args.max_new_tokens)
second_text = second["text"]
print(f"{tag}turn 2: {second['meta_info'].get('completion_tokens')} tokens in {time.time() - t0:.1f}s, finish={second['meta_info'].get('finish_reason')}")
print(f"{tag}turn 2 text tail: {tail(second_text, 500)}")
used = "302,971" in second_text or "302971" in second_text.replace(",", "")
print(f"{tag}RESULT: append_ok forced_call={forced} answer_uses_tool_result={used}")
