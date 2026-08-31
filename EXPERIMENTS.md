# Experiments behind verl #7617 / #7619 and sglang-omni #1840

Everything here ran on tokenizer files only (no weights) unless marked GPU. Scripts
clone the repositories they compare, so each one is reproducible on its own.

## verl #7617 — generation-prompt delta in incremental tokenization

| question | script | write-up / where the numbers went |
|---|---|---|
| Can the generation prompt come out of the **last append group's render** (`add_generation_prompt=True` on that render) instead of two full-history renders? | `verl_fold_probe.py` | [`generation_prompt_fold_experiment.md`](generation_prompt_fold_experiment.md) — 262 template-rendered cases, 0 differences; posted on [#7617](https://github.com/verl-project/verl/issues/7617#issuecomment-5473497404) |
| Is the delta **constant across turns** per family, and does it depend on the final role? | `verl_cacheability_check.py` | table below; the role dependence (DeepSeek emits nothing after a tool output) is posted on [#7617](https://github.com/verl-project/verl/issues/7617#issuecomment-5473506807) |
| Does **caching** the delta (keyed by final role + tools, revalidated) reproduce main's token ids, and how much does it save? | `verl_cache_bench.py` | table in the [#7619](https://github.com/verl-project/verl/pull/7619) description |
| What does each template's `add_generation_prompt` guard actually read? | `verl_template_snippets.py` | Jinja table in [`generation_prompt_experiment.md`](generation_prompt_experiment.md) |
| Why do the three gpt-oss renders disagree on a tool-final append? | `verl_gptoss_detail.py` | decoded diff in both write-ups: the tool path is hand-built, and tool content is not JSON-encoded |
| Earlier: bounded pseudo-tail render vs full-history render, 28 scenarios × 4 Qwen models, 50/100/200-turn timings | `verl_template_check.py` | superseded by the cache design; numbers were in the first version of the #7619 description |

### Delta constant across 20 tool turns? (`verl_cacheability_check.py`, main @a0bd149)

| family | model | constant over 20 turns | same for tool / user / system final |
|---|---|---|---|
| qwen25 | Qwen2.5-7B-Instruct | yes | yes |
| qwen3 | Qwen3-8B | yes | yes |
| qwen35 | Qwen3.5-9B | yes | no — the template refuses a non-leading system message |
| minimaxm2 | MiniMax-M2 | yes | yes |
| glm47 | GLM-4.7 | yes | yes |
| gptoss | gpt-oss-20b | yes | yes |
| deepseek | DeepSeek-V3.2-Exp | yes per role | **no** — `<｜Assistant｜></think>` after a user turn, empty after a tool output |
| default | Qwen3-8B via the base builder | yes | yes |

Not loadable: GLM-5, Gemma-4/3 (gated), DeepSeek-V4 (gated); VL builders need a processor.

### Cache vs main, 100-turn tool loops (`verl_cache_bench.py`)

| builder | model | main | cache branch | last turn | ids |
|---|---|---|---|---|---|
| qwen3 | Qwen3-8B | 6.31s | 0.19s | 147 ms → 1.5 ms | identical |
| qwen35 | Qwen3.5-9B | 6.16s | 0.25s | 117 ms → 1.9 ms | identical |
| glm47 | GLM-4.7 | 4.50s | 0.30s | 91 ms → 1.4 ms | identical |
| minimaxm2 | MiniMax-M2 | 4.94s | 0.31s | 98 ms → 1.4 ms | identical |
| gptoss | gpt-oss-20b | 5.55s | 0.22s | 111 ms → 0.3 ms | identical |
| deepseek | DeepSeek-V3 | 4.80s | 0.22s | 96 ms → 0.7 ms | identical |
| default | Qwen3-8B | 6.14s | 0.35s | 112 ms → 1.6 ms | identical |

## sglang-omni #1840 — output budget for short audio (GPU)

`bench_975.sh` (1× H100): the PR's base commit vs the PR build on one dependency
stack, 13 clips × 3 requests. 11 clips byte-identical at the same latency; the two
where the model loops on repeated `[t][t][S01]` markers are capped at 128 tokens
(1.8× and 2.6× faster). Table posted on
[#1840](https://github.com/sgl-project/sglang-omni/pull/1840#issuecomment-5473412278).
