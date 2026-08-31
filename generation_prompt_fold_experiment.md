# Folding the generation prompt into the last append group — verl #7617

**Question.** `tokenize_non_assistant_incremental_messages` renders every append
group against a bounded synthetic prefix, then renders the *full* history twice
(`add_generation_prompt` False / True) just to obtain the generation prompt. Can
those two full-history renders be dropped by rendering the **last append group**
with `add_generation_prompt=True`, so one render yields `append group + generation
prompt`?

**Answer.** Yes. On the trajectories `tool_agent_loop` actually produces, the folded
render gives exactly the token ids of today's path on every template that renders
them — 216 cases across 9 templates, 0 differences — and, wherever the naive
full-history diff is available as an independent reference, both agree with it too.
Gemma 4, whose generation prompt depends on what the previous message was, is
reproduced as well. The only mismatches are gpt-oss tool groups, which never go
through the chat template (`GptOssContinuousTokenBuilder._tokenize_tool_group`
builds the string by hand), so there is nothing for the flag to act on.

Script: [`verl_fold_probe_harness.py`](https://github.com/ruiling-smartbear/sglang-omni/blob/bench/975/verl_fold_probe_harness.py)
(upstream main @8e4a572, transformers 5.16.1 — Gemma 4's tokenizer needs ≥ 5 to
load — tokenizer files only, CPU). An earlier variant,
[`verl_fold_probe.py`](https://github.com/ruiling-smartbear/sglang-omni/blob/bench/975/verl_fold_probe.py),
also exercised user / system appends and mixed groups that the API allows but no
built-in loop produces; it reached the same conclusion and is kept at the end as
an appendix.

## The trajectories

Built the way `verl/experimental/agent_loop/tool_agent_loop.py` builds them: an
optional system prompt, the user prompt, then per turn an assistant message from
`_build_assistant_message` (empty content, `tool_calls` with `arguments` as the
JSON object `OpenAIFunctionCallSchema.model_dump()` returns, an `id`) and one tool
message per call carrying only `content` and `tool_call_id`. The append under
test is the last tool group — the only append group that loop ever produces.

24 cases per model: system prompt on/off × 0 / 10 / 50 prior tool turns × 1 or 2
parallel calls in the final assistant turn × `enable_thinking` default / False.
Templates without an `enable_thinking` switch ignore the kwarg; their two halves
are the same renders, so every model is scored on the same 24 inputs.

```json
// system prompt, 0 prior turns, 1 call
{
 "previous": [
  {
   "role": "system",
   "content": "You are a helpful agent. Use tools to answer."
  },
  {
   "role": "user",
   "content": "Find the population of Pittsburgh and its ten largest suburbs."
  },
  {
   "role": "assistant",
   "content": "",
   "tool_calls": [
    {
     "type": "function",
     "function": {
      "name": "lookup",
      "arguments": {
       "q": "0-0"
      }
     },
     "id": "call-0-0"
    }
   ]
  }
 ],
 "appended": [
  {
   "role": "tool",
   "content": "Pittsburgh had 302,971 residents at the latest census estimate.",
   "tool_call_id": "call-0-0"
  }
 ],
 "tools": [
  {
   "type": "function",
   "function": {
    "name": "lookup",
    "description": "Look up a population figure.",
    "parameters": {
     "type": "object",
     "properties": {
      "q": {
       "type": "string"
      }
     },
     "required": [
      "q"
     ]
    }
   }
  }
 ]
}
// no system prompt, 0 prior turns, 2 parallel calls
{
 "previous": [
  {
   "role": "user",
   "content": "Find the population of Pittsburgh and its ten largest suburbs."
  },
  {
   "role": "assistant",
   "content": "",
   "tool_calls": [
    {
     "type": "function",
     "function": {
      "name": "lookup",
      "arguments": {
       "q": "0-0"
      }
     },
     "id": "call-0-0"
    },
    {
     "type": "function",
     "function": {
      "name": "lookup",
      "arguments": {
       "q": "0-1"
      }
     },
     "id": "call-0-1"
    }
   ]
  }
 ],
 "appended": [
  {
   "role": "tool",
   "content": "Pittsburgh had 302,971 residents at the latest census estimate.",
   "tool_call_id": "call-0-0"
  },
  {
   "role": "tool",
   "content": "Second lookup: 1,244,000 in the metro area.",
   "tool_call_id": "call-0-1"
  }
 ],
 "tools": [
  {
   "type": "function",
   "function": {
    "name": "lookup",
    "description": "Look up a population figure.",
    "parameters": {
     "type": "object",
     "properties": {
      "q": {
       "type": "string"
      }
     },
     "required": [
      "q"
     ]
    }
   }
  }
 ]
}
```

With `prior_turns` = 10 or 50, the same assistant-tool-call / tool-response pair (one call, `arguments` `{"q": "<turn>-0"}`, ids `call-<turn>-0`) is repeated that many times between the user prompt and the final assistant turn.


Three quantities per case:

| name | how it is produced |
|---|---|
| `today` | upstream main as is: bounded render of the tool group + `_tokenize_generation_prompt_delta` (two full-history renders) |
| `folded` | the same builder, except the last append group's render passes `add_generation_prompt=True` and the two full-history renders are skipped |
| `full-history diff` | `render_delta_token_id(previous, appended, add_generation_prompt=True)`: render the whole conversation twice and take the suffix — an independent reference, available only where the template is prefix-stable |

## Results

| builder | model | thinking switch | folded == today | today == full-history diff | folded == full-history diff |
|---|---|---|---|---|---|
| qwen | Qwen2-7B-Instruct | no | 24/24 | 24/24 | 24/24 |
| qwen25 | Qwen2.5-7B-Instruct | no | 24/24 | 24/24 | 24/24 |
| qwen3 | Qwen3-8B | yes | 24/24 | n/a¹ | n/a¹ |
| qwen35 | Qwen3.5-9B | yes | 24/24 | 24/24 | 24/24 |
| minimaxm2 | MiniMax-M2 | no | 24/24 | 24/24 | 24/24 |
| glm47 | GLM-4.7 | yes | 24/24 | 24/24 | 24/24 |
| gemma4 | google/gemma-4-12b-it | yes | 24/24 | 24/24 | 24/24 |
| gptoss | gpt-oss-20b | no | 0/24² | 0/24² | 0/24² |
| deepseek | DeepSeek-V3.2-Exp | no | –³ | –³ | –³ |
| default | SmolLM3-3B | yes | 24/24 | 24/24 | 24/24 |
| default | Qwen3-8B via the base builder | yes | 24/24 | n/a¹ | n/a¹ |

¹ Qwen3-8B's template re-renders earlier assistant turns when a message is
appended, so rendering `previous` is not a token prefix of rendering
`previous + appended` and the naive diff raises (`Continuous Token token-id suffix
diff failed`). That is the reason the bounded synthetic-prefix renders exist. The
folded render still matches today's path on all 24.

² gpt-oss: `_tokenize_tool_group` formats the tool response with an f-string and
calls `tokenizer.encode`, so `add_generation_prompt` has nothing to act on and the
folded output is today's output minus the trailing `<|start|>assistant`. In an
implementation that builder appends the constant itself. Today's path also never
matches the full-history render for gpt-oss: the template JSON-encodes tool content
(`<|message|>"..."<|end|>`) while `_format_tool_response` writes it raw. That is
pre-existing and independent of this change.

³ DeepSeek-V3.2-Exp: every tool append fails on main today.
`_synthetic_assistant_for_tools` sets `"arguments": {}` and the template
concatenates `arguments` as a string (`TypeError: can only concatenate str (not
"dict") to str` when rendered through jinja2 directly, see
`verl_deepseek_template.py`). The harness passes a mapping as well, so the
full-history render of a real trajectory fails the same way.

Tail of the folded output — the rendered generation prompt — for reference: Qwen
`<|im_start|>assistant\n` (plus `<think>\n\n</think>\n\n` with
`enable_thinking=False`), Qwen3.5 `<|im_start|>assistant\n<think>\n`, MiniMax-M2
`]~b]ai\n<think>\n`, GLM-4.7 `<|assistant|><think>` / `<|assistant|></think>`,
gpt-oss `<|start|>assistant`, Gemma 4 after a tool response: nothing.

**Gemma 4** is the row that tests the assumption hardest. Its template emits the
generation prompt conditionally:

```
{%- if add_generation_prompt -%}
  {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
    {{- '<|turn>model\n' -}}
    {%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}
  {%- elif ns.prev_message_type ...
```

i.e. nothing after a tool response, which is what verl's `Gemma4ContinuousTokenBuilder`
hard-codes today. The last append group carries that state, so the folded render
reproduces it, and the full-history render agrees: 24/24 on all three columns.

## Takeaways

1. The generation prompt can be produced by the last append group's render. No
   full-history render is needed, no cache, no re-validation on the hot path.
2. Two rules for the implementation: only the last append group of a call gets
   the flag, and builders that bypass the template for a group (gpt-oss tool
   responses) append the generation prompt themselves.
3. The assumption behind it is the one the bounded synthetic-prefix renders
   already make — the tail of the conversation determines what the template emits
   next. Every template measured satisfies it, Gemma 4 included. For templates
   nobody has measured, a one-time check on the first incremental call (folded
   suffix vs the two full-history renders, then never again unless they disagree)
   keeps the fallback without paying for it per turn.
4. Two pre-existing issues surfaced along the way and are worth their own reports:
   the DeepSeek `arguments: {}` shape above, and gpt-oss tool content being written
   raw by `_format_tool_response` while the template JSON-encodes it.

## Appendix — the earlier, API-shaped matrix

### What was compared (earlier run)

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

### Results (earlier run)

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

### Takeaways of that run

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
