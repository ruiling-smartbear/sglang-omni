# Folding the generation prompt into the last append group — verl #7617

**Question.** `tokenize_non_assistant_incremental_messages` renders every append
group against a bounded synthetic prefix, then renders the *full* history twice
(`add_generation_prompt` False / True) just to obtain the generation prompt. Can
those two full-history renders be dropped by rendering the **last append group**
with `add_generation_prompt=True`, so one render yields `append group + generation
prompt`?

**Answer.** Yes. On upstream `main` @8e4a572 with real tokenizers, the folded
render produces exactly the same token ids as today's path in every case that
today's path can render — 0 differences over 8 models, 240 cases. The only
mismatches are gpt-oss tool groups, which never go through the chat template
(`GptOssContinuousTokenBuilder._tokenize_tool_group` builds the string by hand),
so there is nothing for the flag to act on; that builder would append its constant
`<|start|>assistant` itself.

Script: [`verl_fold_probe.py`](https://github.com/ruiling-smartbear/sglang-omni/blob/bench/975/verl_fold_probe.py).
Tokenizer files only, no weights, CPU.

## What was compared

`current` = upstream main as is: bounded render of each append group +
`_tokenize_generation_prompt_delta` (two full-history renders).

`folded` = the same builder with two changes: the render of the last append group
passes `add_generation_prompt=True`, and the two full-history renders are skipped.
Earlier append groups in the same call are rendered unchanged. The prefix side of
the suffix diff still uses `add_generation_prompt=False`.

Cases per model: tools on/off × prior history of 0 or 10 tool turns × five
appends — `[tool]`, `[tool, tool]` (one group, two responses to a two-call
assistant turn), `[user]`, `[system]`, `[tool] + [user]` (two groups, only the
second gets the flag) — × `enable_thinking` False/default where the template has
that switch.

Conversation (prior turn pattern and the `[tool]` append shown):

```json
{"previous": [
   {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
   {"role": "user", "content": "Find the population of Pittsburgh."},
   {"role": "assistant", "content": "", "tool_calls": [{"id": "prior-0", "type": "function",
       "function": {"name": "lookup", "arguments": {"q": "0"}}}]},
   {"role": "tool", "content": "Pittsburgh had 302,971 residents at the latest census estimate.",
       "tool_call_id": "prior-0", "name": "lookup"},
   "... repeated for prior_turns ...",
   {"role": "assistant", "content": "", "tool_calls": [{"id": "call-a", "type": "function",
       "function": {"name": "lookup", "arguments": {"q": "98"}}}]}],
 "appended": [
   {"role": "tool", "content": "Pittsburgh had 302,971 residents at the latest census estimate.",
       "tool_call_id": "call-a", "name": "lookup"}],
 "tools": [{"type": "function", "function": {"name": "lookup", "description": "Look up a population figure.",
       "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]}
```

## Results

| builder family | model | identical | differ | not rendered by main either |
|---|---|---|---|---|
| qwen | Qwen2-7B-Instruct | 20/20 | 0 | 0 |
| qwen25 | Qwen2.5-7B-Instruct | 20/20 | 0 | 0 |
| qwen3 | Qwen3-8B | 40/40 | 0 | 0 |
| qwen35 | Qwen3.5-9B | 32/32 | 0 | 8 — the template refuses a system message that is not first, on both paths |
| minimaxm2 | MiniMax-M2 | 20/20 | 0 | 0 |
| glm47 | GLM-4.7 | 40/40 | 0 | 0 |
| gptoss | gpt-oss-20b | 8/8 template-rendered | 12, all appends containing a tool group | 0 |
| deepseek | DeepSeek-V3.2-Exp | 2/2 | 0 | 18 — see below |
| default | SmolLM3-3B | 40/40 | 0 | 0 |
| default | Qwen3-8B (base builder) | 40/40 | 0 | 0 |

Tail of the folded output, i.e. the rendered generation prompt, for reference:
Qwen `<|im_start|>assistant\n` (`...<think>\n\n</think>\n\n` with `enable_thinking=False`),
Qwen3.5 `<|im_start|>assistant\n<think>\n`, MiniMax-M2 `]~b]ai\n<think>\n`,
GLM-4.7 `<|assistant|><think>` / `<|assistant|></think>`, gpt-oss `<|start|>assistant`.

**gpt-oss.** The 12 differing cases are every append that contains a tool group.
`_tokenize_tool_group` there formats the response with an f-string and calls
`tokenizer.encode`, so the flag has nothing to act on and the folded output is
`current` minus the trailing `<|start|>assistant`. The `[tool] + [user]` rows differ
for the same reason (the harness counts template renders to find the last group,
and the tool group made none). In an implementation the builder appends its
constant generation prompt after the hand-built string; user/system appends, which
do go through the template, are identical (8/8).

**DeepSeek-V3.2-Exp.** 18 cases do not render on `main` today, independently of
folding: tool appends fail inside the template because
`_synthetic_assistant_for_tools` sets `"arguments": {}` and this template
concatenates `arguments` as a string (`TypeError: can only concatenate str (not
"list")`), the same error hits the full-history render whenever the history holds
a tool call with mapping arguments, and system appends fail the suffix diff. The
two cases both paths render (user append, no prior tool turns) are identical.

Not loadable here: GLM-5 (tokenizer class not in this transformers), Gemma
(gated), DeepSeek-V4 (gated); VL builders need a processor.

## Takeaways

1. The generation prompt can be produced by the last append group's render. No
   full-history render is needed, no cache, no re-validation on the hot path.
2. Two rules for the implementation: only the last append group of a call gets
   the flag, and builders that bypass the template for a group (gpt-oss tool
   responses) append the generation prompt themselves.
3. The assumption behind it is the same one the bounded synthetic-prefix renders
   already make — the tail of the conversation determines what the template emits
   next. Every template measured satisfies it. For templates nobody has measured,
   a one-time check on the first incremental call (folded suffix vs the two
   full-history renders, then never again unless they disagree) keeps the fallback
   without paying for it per turn.
4. Two pre-existing issues surfaced along the way and are worth their own reports:
   the DeepSeek `arguments: {}` shape above, and gpt-oss tool content being written
   raw by `_format_tool_response` while the template JSON-encodes it.
