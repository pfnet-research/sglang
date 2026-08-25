#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PATH [RESULT_DIR]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="$1"
MODEL_REVISION="${MODEL_REVISION:-}"
RESULT_DIR="${2:-${TMPDIR:-/tmp}/plamo3-mmlu-results}"
DATA_DIR="${MMLU_DATA_DIR:-${SCRIPT_DIR}/data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
NTRAIN="${NTRAIN:-5}"
NSUB="${NSUB:-60}"
PARALLEL="${PARALLEL:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
HF_RESULT="${RESULT_DIR}/mmlu-hf-${RUN_ID}.jsonl"
SGLANG_RESULT="${RESULT_DIR}/mmlu-sglang-${RUN_ID}.jsonl"
SERVER_LOG="${RESULT_DIR}/sglang-server-${RUN_ID}.log"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RESULT_DIR}"

if [[ ! -d "${DATA_DIR}/dev" || ! -d "${DATA_DIR}/test" ]]; then
  archive="$(mktemp "${TMPDIR:-/tmp}/mmlu-data.XXXXXX.tar")"
  cleanup_archive() {
    rm -f "${archive}"
  }
  trap cleanup_archive EXIT
  mkdir -p "${DATA_DIR}"
  curl --fail --location \
    https://people.eecs.berkeley.edu/~hendrycks/data.tar \
    --output "${archive}"
  tar -xf "${archive}" -C "${DATA_DIR}" --strip-components=1
  cleanup_archive
  trap - EXIT
fi

echo "Running Hugging Face MMLU with model: ${MODEL_PATH}"
hf_args=(
  --model-path "${MODEL_PATH}"
  --data-dir "${DATA_DIR}"
  --ntrain "${NTRAIN}"
  --nsub "${NSUB}"
  --output "${HF_RESULT}"
)
if [[ -n "${MODEL_REVISION}" ]]; then
  hf_args+=(--revision "${MODEL_REVISION}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/bench_hf.py" "${hf_args[@]}"

base_url="http://${HOST}:${PORT}"
if curl --fail --silent "${base_url}/model_info" >/dev/null 2>&1; then
  echo "A server is already responding at ${base_url}; choose another PORT." >&2
  exit 1
fi

server_pid=""
cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting SGLang server at ${base_url}"
server_args=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --dtype bfloat16
  --attention-backend triton
  --sampling-backend pytorch
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --host "${HOST}"
  --port "${PORT}"
)
if [[ -n "${MODEL_REVISION}" ]]; then
  server_args+=(--revision "${MODEL_REVISION}")
fi
"${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}" \
  >"${SERVER_LOG}" 2>&1 &
server_pid=$!

server_ready=0
for _ in $(seq 1 180); do
  if curl --fail --silent "${base_url}/model_info" >/dev/null 2>&1; then
    server_ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "SGLang server exited before becoming ready. See ${SERVER_LOG}." >&2
    exit 1
  fi
  sleep 2
done

if [[ "${server_ready}" -ne 1 ]]; then
  echo "Timed out waiting for SGLang server. See ${SERVER_LOG}." >&2
  exit 1
fi

echo "Running SGLang MMLU with model: ${MODEL_PATH}"
sglang_args=(
  --data_dir "${DATA_DIR}"
  --ntrain "${NTRAIN}"
  --nsub "${NSUB}"
  --parallel "${PARALLEL}"
  --host "${HOST}"
  --port "${PORT}"
  --backend srt
  --model-path "${MODEL_PATH}"
  --result-file "${SGLANG_RESULT}"
)
if [[ -n "${MODEL_REVISION}" ]]; then
  sglang_args+=(--revision "${MODEL_REVISION}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/bench_sglang.py" "${sglang_args[@]}"

cleanup_server
server_pid=""
trap - EXIT INT TERM

echo "Hugging Face result: ${HF_RESULT}"
tail -n 1 "${HF_RESULT}"
echo "SGLang result: ${SGLANG_RESULT}"
tail -n 1 "${SGLANG_RESULT}"
