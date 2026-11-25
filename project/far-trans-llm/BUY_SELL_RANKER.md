# Homework 2: BUY/SELL prompt ablation

**Question:** Does showing historical SELL operations separately from BUY
operations affect purchase ranking? Compare the new ranker with Latest Buy or
Sell. To isolate the effect of the prompt change, also compare with the
provided BUY-only prompt using the **same** model, candidates, settings, and
fallback. The comparison with the classical baseline alone cannot isolate the
effect of adding SELL evidence.

## Implementation

- **Reused:** `ranker.py` supplies the original BUY prompt, response parser,
  ranking validation, logging, and invalid-output handling. `runtime.py`
  supplies real local Transformers inference and records raw completions,
  token counts, inference time, and cache usage. `data_adapter.py` calls the
  starter's data filtering, all four classical baselines, and scorer. The
  starter's plugin/response interface is not required: the provided LLM runner
  already delegates these same operations to the starter.
- **Original change:** [`buy_sell_ranker.py`](buy_sell_ranker.py) preserves all
  BUY fields and adds a separate list of up to 15 recent SELL events (date and
  ISIN). The prompt says a SELL is neither a purchase nor an automatic dislike.
  No transaction values, holdings, intraday order, or future rows are inferred.
  Events come from the starter's pre-cutoff scan; it excludes duplicate rows
  and reconstructed initial positions. Empty SELL history is an empty list.
- **Fixed comparison:** The wrapper checks the saved protocol and CSV hashes,
  then uses the saved 50 CBF candidates in the original order for each user.
  The protocol is 100 warm users, model input strictly before `2021-06-01`,
  future purchase labels in `[2021-06-01, 2021-07-01)`, and repeats allowed.
  Full evaluations retain all four classical methods and the matched
  `candidate_order_first10` control. An invalid LLM output uses that same
  first-ten fallback and records the raw output and error; the runner does not
  retry or drop cases.

The prompt-only check matched all 100 users and their ordered candidates to the
saved baseline, found 61 users with a SELL history, and confirmed every dated
event preceded the cutoff. The 100 prompts use 4,746–7,565 input tokens with
the pinned tokenizer, all within the context budget including the generation
allowance. All 400 classical rankings still match the saved baseline rankings.
The matched BUY-only payloads are identical after removing the new SELL field.
The prompts, transaction-history audit, token-budget check, and resolved package
versions are saved locally under `outputs/hw2_buy_sell_prompts/`; matched control
prompts are under `outputs/hw2_buy_only_prompts/`.

## Model and reproduction

[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0),
revision `c1899de289a04d12100db370d81485cdf75e47ca`, is the smallest Qwen3
checkpoint in the provided Transformers model family. The pinned
`model.safetensors` is 1,503,300,328 bytes and is required only on first
download. The local machine is an Apple M1 (8 GiB RAM, macOS 13.5).
The smoke configuration uses Python 3.12.4, PyTorch 2.11.0, and Transformers
4.57.6, with CPU float32, four PyTorch threads, greedy decoding, thinking off,
an 8,192-token context budget, and at most 512 generated tokens. Model
repository code is not executed (`trust_remote_code=False`).

From `project/far-trans-llm`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Optional prompt-only export: no inference, no scores.
.venv/bin/python buy_sell_ranker.py --prepare-only \
  --data-dir ../far-trans/data/session1/FAR-Trans \
  --output-dir outputs/hw2_buy_sell_prompts

# One real generation; add --allow-download only if the weights are not cached.
.venv/bin/python buy_sell_ranker.py --smoke --device cpu \
  --data-dir ../far-trans/data/session1/FAR-Trans \
  --output-dir outputs/hw2_buy_sell_smoke
```

The smoke user is selected from the frozen cohort by past activity alone: the
first user with both pre-cutoff BUY and SELL records. The smoke test writes the
prompt, runtime and model metadata (`experiment.json`), and, once generation
runs, the raw generation (`generations.jsonl`) and the validation result
(`final_responses.jsonl`). It counts as passing only after a fresh local response
passes the ranking validator without fallback. It does not compute nDCG.
Keep these user-level artifacts local and use a fresh directory for reruns.

After a valid smoke test, request `--full` in a fresh directory, first with
`--history-mode buy-sell` and then with `--history-mode buy-only` in a separate
directory. Report both nDCG@10 means, the number of invalid/fallback cases,
coverage without fallback, and elapsed time. No full run is implicit.
