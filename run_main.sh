#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
DATA_ROOT="${DATA_ROOT:-${data_root:-$HOME/Data/new_extracted_nuscenes_val}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results-inference/$(date +%Y%m%d-%H%M%S)}"
GT_PATH="${GT_PATH:-${OCC3D_PATH:-${occ3d_path:-}}}"
: "${GT_PATH:?Set GT_PATH (or OCC3D_PATH) to the occupancy GT directory}"
read -r -a variants <<< "${VARIANTS:-lidar camera_frame}"
export TTOCC_REX_BATCH_INFER="${TTOCC_REX_BATCH_INFER:-1}"
export TTOCC_SAM_BOX_BATCH="${TTOCC_SAM_BOX_BATCH:-1}"
export TTOCC_NN_WORKERS="${TTOCC_NN_WORKERS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Optional dependencies are appended by train.py after active-environment Torch.
# TTOCC_EXTRA_SITE_PACKAGES=/path/to/other/site-packages
extra=()
if [[ -n "${SCENES:-}" ]]; then
    read -r -a scenes <<< "$SCENES"
    extra+=(--scenes "${scenes[@]}")
fi
if [[ -n "${FEATURE_ROOT:-}" ]]; then extra+=(--feature-root "$FEATURE_ROOT"); fi
exec python -B utils/run_inference.py \
    --data-root "$DATA_ROOT" --gt-path "$GT_PATH" --output-root "$OUTPUT_ROOT" \
    --variants "${variants[@]}" --max-frames "${MAX_FRAMES:-0}" \
    --occ-setting "${OCC_SETTING:-Occ3D}" \
    --semantic-prefix "${SEMANTIC_PREFIX:-rexomni}" --depth-prefix "${DEPTH_PREFIX:-mapanything}" \
    --openseed-text-is-eval "${OPENSEED_TEXT_IS_EVAL:-1}" \
    --openseed-semantic-output "${OPENSEED_SEMANTIC_OUTPUT:-sem_seg}" \
    --openseed-instance-mode "${OPENSEED_INSTANCE_MODE:-panoptic}" \
    --map-model "${MAPANYTHING_MODEL_PATH:-facebook/map-anything}" \
    --map-revision "${MAPANYTHING_REVISION:-a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa}" \
    --rex-model "${REXOMNI_MODEL_PATH:-IDEA-Research/Rex-Omni}" \
    --rex-revision "${REXOMNI_REVISION:-0e5693d24657f6c0e091008dd6809bb1bd28988c}" \
    --fusion "${USE_FUSION:-1}" --verbose "${VERBOSE:-1}" "${extra[@]}" "$@"
