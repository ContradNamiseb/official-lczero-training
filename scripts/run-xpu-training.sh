#!/usr/bin/env bash
# Verify and launch KDA training on an Intel GPU via Intel Extension for
# TensorFlow. Run inside native Ubuntu 22.04 or WSL2, not Windows.
set -euo pipefail

config="tf/configs/kda-hybrid-xpu.yaml"
test_steps=500
num_test_positions=512
detailed_summaries=0
saved_model_checkpoints=0
run_setup=0
verify_only=0

usage() {
    cat <<'USAGE'
Usage: run-xpu-training.sh [options]

  --config PATH              Training config (default tf/configs/kda-hybrid-xpu.yaml)
  --test-steps N             Steps between test passes (default 500)
  --num-test-positions N     Positions per test pass (default 512)
  --detailed-summaries       Keep full TensorBoard histograms
  --saved-model-checkpoints  Keep SavedModel exports at every checkpoint
  --setup                    Run scripts/setup-xpu.sh first
  --verify-only              Check environment and config, then exit
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config) config="$2"; shift 2 ;;
        --test-steps) test_steps="$2"; shift 2 ;;
        --num-test-positions) num_test_positions="$2"; shift 2 ;;
        --detailed-summaries) detailed_summaries=1; shift ;;
        --saved-model-checkpoints) saved_model_checkpoints=1; shift ;;
        --setup) run_setup=1; shift ;;
        --verify-only) verify_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv-xpu/bin/python"

if [ "${run_setup}" -eq 1 ] || [ ! -x "${python_bin}" ]; then
    "${repo_root}/scripts/setup-xpu.sh"
fi

if [ ! -x "${python_bin}" ]; then
    echo "XPU environment not found. Run scripts/setup-xpu.sh first." >&2
    exit 1
fi

case "${config}" in
    /*) config_path="${config}" ;;
    *) config_path="${repo_root}/${config}" ;;
esac

if [ ! -f "${config_path}" ]; then
    echo "Training config not found: ${config_path}" >&2
    exit 1
fi

"${python_bin}" -c "import tensorflow as tf; d=tf.config.list_physical_devices('XPU'); print('TensorFlow', tf.__version__); print('XPUs:', d); assert d, 'Intel Extension for TensorFlow did not expose an XPU device'"

"${python_bin}" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); t=c['training']; m=c['model']; assert t['batch_size'] % t['num_batch_splits'] == 0; print('Config:', c['name']); print('Precision:', t.get('precision', 'single')); print('Mixers:', m['encoder_mixer_pattern']); print('Microbatch:', t['batch_size'] // t['num_batch_splits'])" "${config_path}"

if [ "${verify_only}" -eq 1 ]; then
    echo "XPU environment and config verification passed."
    exit 0
fi

"${python_bin}" -c "import glob,os,sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); d=c['dataset']; paths=d.get('input_train', []) + d.get('input_test', []); placeholders=[p for p in paths if '/path/to/' in p]; assert not placeholders, 'Replace placeholder dataset paths: ' + ', '.join(placeholders); fast=d.get('fast_chunk_loading', True); missing=[p for p in paths if not (os.path.isdir(p.replace('*/','')) if fast else glob.glob(p))]; assert not missing, 'Dataset paths not found: ' + ', '.join(missing); print('Dataset roots verified:', len(paths))" "${config_path}"

train_args=(
    ./train.py
    --cfg "${config_path}"
    --test-steps "${test_steps}"
    --num-test-positions "${num_test_positions}"
)
if [ "${detailed_summaries}" -eq 0 ]; then
    train_args+=(--disable-detailed-summaries)
fi
if [ "${saved_model_checkpoints}" -eq 0 ]; then
    train_args+=(--disable-saved-model-checkpointing)
fi

cd "${repo_root}/tf"
echo "Starting XPU training. Press Ctrl+C to stop."
echo "Runtime monitoring: test every ${test_steps} steps over ${num_test_positions} positions."
exec "${python_bin}" "${train_args[@]}"
