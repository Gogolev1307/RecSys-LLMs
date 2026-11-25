# Homework 2: run checklist and report outline

## Run order

1. Let the existing CPU smoke test finish. From `project/far-trans-llm`, read
   the current results and measured generation time without restarting anything:

   ```bash
   .venv/bin/python compare_hw2.py --smoke-dir outputs/hw2_buy_sell_smoke
   ```

   Proceed only if the status is `valid_real_inference`: one fresh model
   response, ten unique in-pool ISINs after validation, and no fallback.
   `generations.jsonl` records generation seconds separately from download,
   loading and preparation time in `experiment.json`. The estimate multiplies
   that one measurement by 100; individual token lengths vary. Leave time
   for grading and artifact checks, not just inference.

2. If the CPU estimate fits the budget, run **only** the mandatory BUY/SELL
   variant in a fresh directory. This command is prepared, not executed:

   ```bash
   .venv/bin/python buy_sell_ranker.py --full --history-mode buy-sell \
     --device cpu --model Qwen/Qwen3-0.6B \
     --revision c1899de289a04d12100db370d81485cdf75e47ca \
     --on-invalid fallback \
     --data-dir ../far-trans/data/session1/FAR-Trans \
     --baseline-dir ../far-trans/outputs/hw2_baseline \
     --output-dir outputs/hw2_buy_sell_full
   ```

   The completed smoke run will have cached the weights, so the local command
   deliberately omits `--allow-download`. The BUY-only run is optional.

3. When full evaluation completes, produce the four-decimal table and check
   protocol, baseline rankings, and failure consistency:

   ```bash
   .venv/bin/python compare_hw2.py \
     --llm-dir outputs/hw2_buy_sell_full \
     --smoke-dir outputs/hw2_buy_sell_smoke
   ```

   Keep the `experiment.json`, `run_manifest.json`, `metrics.json`,
   `predictions.csv`, `per_user.json`, prompts, and response/generation logs
   locally. Do not publish user-level data or the full hashes in the main table.

## Fastest GPU alternative: Colab

For copy/paste cells using the portable ZIP, see [the Colab guide](COLAB_HW2.md).

If the local estimate exceeds the remaining time, select a **GPU** runtime in
Colab. Copy the current working source (including the new, not necessarily
committed files), the four original CSVs, and the original
`far-trans/outputs/hw2_baseline/` directory to the session under the same
relative paths. Use a private upload or Drive location and keep user-level
artifacts private.

Use the Colab CUDA PyTorch already installed, pin `transformers==4.57.6`, and
check `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported(including_emulation=False)`
before running. Prefer an L4 or A100 with native bfloat16. For a T4, add
`--cuda-dtype float16` to both GPU runs; only numerical precision changes.
Do not silently fall back to CPU. Record GPU model, PyTorch/CUDA
versions, and the dtype. Copy the local
command, change `.venv/bin/python` to `python`, change `--device cpu` to
`--device cuda`, and add `--allow-download` for the same pinned checkpoint on
first use in Colab. First try `--smoke` with a fresh GPU smoke directory; then
run `--full` if it passes and its estimate fits. Move the completed outputs
back before the session expires. The device/dtype can differ from the local
smoke, but the checkpoint, users, candidates, prompt, and scoring stay fixed.

## Individual PDF outline (five pages excluding references)

- **Title and Abstract:** Name, team, GitHub repository and experiment
  commit. One sentence each for the question, intervention, and result. Mark
  an unfinished run as incomplete, not as a negative result.
- **Introduction and Related Work:** Motivate labeling past BUY and SELL
  operations separately. Cite FAR-Trans, the provided ranker/LLMRank
  adaptation, and the exact Qwen checkpoint; note that the task is purchase
  prediction, not investment suitability or returns.
- **Method:** Explain the provided runner/parser/scorer and your original
  separate SELL list. Document the exact prompt location, model revision,
  dependencies, hardware, and decoding settings. Describe validation and the
  deterministic first-ten CBF fallback.
- **Experiments:** Describe the frozen 100 warm users, 50 ordered CBF
  candidates, June 1 cutoff, June 1–30 labels, repeat purchases, and binary
  nDCG@10. Insert the table for four baselines plus the ranker, with both means,
  both user counts, elapsed time, and failures. Give no-fallback coverage and
  state that fallback cases remain in the average. **Personally verify** one
  item: for example, trace a saved SELL back to a pre-cutoff source row and
  check the ten returned ISINs against that user's candidate list. Describe
  what *you* checked and found, separately from the agent's automated audits.
- **Discussion:** Compare with Latest Buy or Sell, describe failures and the
  11-user nonempty subset, and avoid attributing a score difference to SELL
  labeling alone without a matched BUY-only ablation. Record the team's actual
  discussion, decisions, and open questions. Distinguish shared work from what
  you personally proposed, implemented, reviewed, or verified; do not fill in
  unconfirmed details. State the next experiment.
- **AI Usage Disclosure and References:** Disclose the actual assistant use
  for code, tests, and reporting support. Distinguish it from the real Qwen
  model used for inference, and identify any checks you performed personally.
  Submit the individual PDF and AI-session export (native JSONL, or a
  documented derivative retaining code, tool calls, and results).
