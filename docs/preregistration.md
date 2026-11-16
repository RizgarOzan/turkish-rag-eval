# Pre-registration: query language and abstention

Committed before any run of this experiment. The commit date is the timestamp.

## Question

When the passages are Turkish but the question is asked in English, do generators still say "I don't know" when the answer isn't in the passages, or do they start making things up?

## Setup (fixed before running)

- Questions: the 58 human-labelled ones, and separately the 242 drafted ones (LLM-drafted, not reviewed).
- Contrast: question in Turkish vs. the same question translated to English. Passages stay Turkish.
- Retrieval: hierarchical + bm25_stem5, k=5, run on the Turkish question only. The English question gets the same five passages, so the only thing that changes is the language of the question. (bm25 matches words, so an English query against Turkish passages would retrieve almost nothing and swamp any effect on the generator.)
- Generators: the same eight as the re-judged groundedness table: claude-cli:claude-haiku-5-5, claude-cli:claude-sonnet-5-5, claude-cli:claude-opus-5-5, nvidia:moonshotai/kimi-k3, nvidia:z-ai/glm-5.3-flash, groq:openai/gpt-oss-120b, groq:openai/gpt-oss-20b, groq:qwen/qwen3.8-27b. Same prompt as the Turkish runs. The Claude CLI can't set temperature; the others run at temperature 0.
- Judge: nvidia:nvidia/nemotron-3-ultra-550b-a55b, temperature 0. (The earlier judge, nemotron-3-super-120b-a12b, was retired by NVIDIA on 2026-10-03; every earlier groundedness run is re-judged with this one in `results/groundedness-ultra/`.) The Turkish side is not re-run: it is the existing `results/groundedness-ultra/` run, so the comparison is per question against those answers.
- Translation: claude-sonnet-5-5 translates each question once, told to keep names, numbers and quoted terms as they are. I read all 58 labelled translations against the Turkish and fix any that change the meaning; the fixes are committed before the run. For the 242 drafted questions I read a random 25 (seed 0) and report how many were wrong, but don't fix the rest. Sonnet is also one of the generators, so it answers questions it translated itself; I note this rather than work around it.

## What I expect

Baseline (Turkish, 58 labelled questions, 33 with the gold span retrieved, 25 without):

| metric | Turkish now | what I expect in English |
|---|---|---|
| answered when the span was retrieved | 30–33 of 33 | 1–2 fewer per model, mostly the weaker models (gpt-oss-20b, gpt-oss-120b, qwen) |
| answer rests on the passages | 96.7–100% | unchanged |
| answer conveys the gold span | 90.3–100% | unchanged, within one answer |
| declined when the span was not retrieved | 19–20 of 25 | unchanged, within ±2 answers for every model |
| hallucinated (span not retrieved) | 0–3 of 25 | unchanged, within ±2 answers |

Why: the model has to match an English question to Turkish text. When the answer is there, that extra step should occasionally make a model miss it and decline, which is why I expect a small drop in the first row and why it should hit the smaller models. When the answer isn't there, I don't see a reason for the question's language to make a model more willing to guess; the Turkish passages are what it has to check either way. So my main prediction for abstention and hallucination is no effect. If English questions turn out to make models guess more, that would be the interesting result and I'd be wrong.

## What would count as no effect

The comparison is paired: the same question, the same five passages, Turkish vs. English. For each model and each metric I count the questions that flip in each direction (e.g. declined in Turkish, answered in English).

- For a single model, a net change of 2 answers or fewer is no effect. On 25 questions one answer is 4 points, and changing the judge alone moved two models by one answer, so smaller differences can't be told apart from noise.
- A net change of 3 or more is reported as a difference, along with an exact McNemar test on the flips. With 25 questions this test has little power, so I don't claim significance for a single model.
- Across models, I call it an effect of language only if at least 6 of the 8 models move the same way. Fewer than that is no effect.

The 242 drafted questions are a secondary check only. If they disagree with the 58, the 58 decide, since nobody has checked the drafted labels.

## What I will report whatever happens

- Every model run, both languages, nulls included, in the same table format as `docs/results.md`.
- The flip counts per model and metric, not only the rates.
- How many of the 58 translations I corrected, and how many of the 25 sampled drafted ones were wrong.
- Any run that failed or was cut short, and why. No model is dropped after seeing its result.
