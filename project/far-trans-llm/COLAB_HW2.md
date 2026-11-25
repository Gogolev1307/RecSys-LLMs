# Colab: full Homework 2 BUY/SELL run

Select **Runtime → Change runtime type → GPU**. Prefer **L4/A100**. Upload
`hw2_colab_bundle.zip` to `/content/` using Colab's Files pane. Keep the
notebook, dataset, baseline contexts and user-level outputs private. The ZIP
contains no credentials, local environment or model weights.

## 1. Extract and check the GPU

```bash
%%bash
set -eu
unzip -q /content/hw2_colab_bundle.zip -d /content/hw2
python -m pip install transformers==4.57.6
python -c 'import torch; assert torch.cuda.is_available(), "Select a GPU runtime"; print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, "native bf16:", torch.cuda.is_bf16_supported(including_emulation=False))'
```

Use Colab's preinstalled CUDA PyTorch. The CPU smoke used PyTorch 2.11.0;
the GPU run logs the actual version, CUDA version, GPU, and precision.

## 2. Run one real GPU smoke test

```bash
%%bash
set -eu
cd /content/hw2/project/far-trans-llm
DTYPE=$(python -c 'import torch; assert torch.cuda.is_available(); print("bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16")')
python buy_sell_ranker.py --smoke --history-mode buy-sell --device cuda --cuda-dtype "$DTYPE" \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --allow-download --on-invalid fallback \
  --data-dir ../far-trans/data/session1/FAR-Trans \
  --baseline-dir ../far-trans/outputs/hw2_baseline \
  --output-dir outputs/hw2_buy_sell_gpu_smoke
python compare_hw2.py --smoke-dir outputs/hw2_buy_sell_gpu_smoke
```

The first run downloads about 1.5 GB of public model weights. L4/A100 use
native bfloat16; a T4 uses float16. This selects only the tensor dtype:
the checkpoint, prompt, greedy generation, token limits, candidate order,
fallback and grading are unchanged. Lower precision can still change
individual completions. The T4 path was checked in code, not measured on
GPU here, so require `valid_real_inference` before starting the full run.

## 3. Run the 100-user experiment and export results

If the GPU smoke passes and its time estimate fits, run the next cell **in
the same Colab runtime**, where the model is already cached:

```bash
%%bash
set -eu
cd /content/hw2/project/far-trans-llm
DTYPE=$(python -c 'import json; e=json.load(open("outputs/hw2_buy_sell_gpu_smoke/experiment.json")); assert e["status"] == "valid_real_inference", e["status"]; print(e["requested_cuda_dtype"])')
python buy_sell_ranker.py --full --history-mode buy-sell --device cuda --cuda-dtype "$DTYPE" \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --on-invalid fallback \
  --data-dir ../far-trans/data/session1/FAR-Trans \
  --baseline-dir ../far-trans/outputs/hw2_baseline \
  --output-dir outputs/hw2_buy_sell_full
python compare_hw2.py --llm-dir outputs/hw2_buy_sell_full \
  --smoke-dir outputs/hw2_buy_sell_gpu_smoke | tee outputs/hw2_comparison.md
zip -qr /content/hw2_buy_sell_results.zip \
  outputs/hw2_buy_sell_full outputs/hw2_buy_sell_gpu_smoke outputs/hw2_comparison.md \
  -x '*/generation_cache/*'
```

Download `/content/hw2_buy_sell_results.zip` from the Files pane before
disconnecting. Use fresh output directory names for intentional reruns.
The optional BUY-only ablation is not required to execute the BUY/SELL run.
