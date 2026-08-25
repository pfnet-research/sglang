## Download data
```
bash download_data.sh
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --nsub 10
```

```
# OpenAI models
python3 bench_sglang.py --backend gpt-3.5-turbo --parallel 8
```

## Reproduce the PLaMo3 Hugging Face/SGLang comparison

Run the full 5-shot MMLU comparison (57 subjects, 14,042 questions) with an
explicit model path:

```bash
PYTHON_BIN=.venv-plamo3/bin/python \
  bash benchmark/mmlu/run_plamo3_hf_sglang.sh \
  pfnet/plamo-3-nict-2604-31b-base
```

The PR #1 measurements use the public, gated checkpoint
`pfnet/plamo-3-nict-2604-31b-base` at Hugging Face revision
`e4ff0386e3aeb53a7d60e4b10ab3589f5d95c343`. Its `config.json` SHA256 is
`ef8bac4be50b04216071bae6610e61ceac5820c44e6c00917fe4d195388c5769`.
It is a 31B BF16 checkpoint with 64 layers and a 256K context length.

The script downloads MMLU data when needed, runs Hugging Face first, starts an
SGLang server with the same settings used for the PR, runs SGLang with
`parallel=64`, and writes both JSONL results under
`${TMPDIR:-/tmp}/plamo3-mmlu-results` by default. The model is gated, so accept
its access conditions and authenticate with `hf auth login` first.
