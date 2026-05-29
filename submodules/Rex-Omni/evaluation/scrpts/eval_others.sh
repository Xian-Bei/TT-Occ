#!/bin/bash

# Generic Box Evaluation Pipeline (HierText / DocLayNet / HumanRef / Dense200)
# Runs inference (inference_text_prompt.py) then evaluates (other_metric.py)

set -e  # Exit on any error

# Defaults
DATASET="HierText"  # Options: HierText, DocLayNet, HumanRef, Dense200, etc
MODEL_PATH="IDEA-Research/Rex-Omni"
IMAGE_ROOT="Mountchicken/Rex-Omni-Eval"
OUTPUT_BASE="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/box_eval"
START_IDX=0
END_IDX=-1
MAX_TOKENS=4096
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))
QUESTION="Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."

# Optional explicit overrides
TEST_JSONL=""
OUTPUT_DIR_OVERRIDE=""
EVAL_TYPE="box_eval"  # Options: box_eval, point_eval, visual_prompt_eval

print_help() {
  echo "Usage: $0 [--dataset NAME] [--model_path PATH] [--image_root DIR] [--start_idx N] [--end_idx N]"
  echo "             [--max_tokens N] [--min_pixels N] [--max_pixels N] [--question STR]"
  echo "             [--test_jsonl PATH] [--output_dir DIR]"
  echo "             [--eval_type box_eval|point_eval|visual_prompt_eval]"
  echo "\nDatasets: HierText, DocLayNet, HumanRef, Dense200, IC15, M6Doc, RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone"
}

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"; shift 2;;
    --model_path)
      MODEL_PATH="$2"; shift 2;;
    --image_root)
      IMAGE_ROOT="$2"; shift 2;;
    --start_idx)
      START_IDX="$2"; shift 2;;
    --end_idx)
      END_IDX="$2"; shift 2;;
    --max_tokens)
      MAX_TOKENS="$2"; shift 2;;
    --min_pixels)
      MIN_PIXELS="$2"; shift 2;;
    --max_pixels)
      MAX_PIXELS="$2"; shift 2;;
    --question)
      QUESTION="$2"; shift 2;;
    --test_jsonl)
      TEST_JSONL="$2"; shift 2;;
    --output_dir)
      OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
    --eval_type)
      EVAL_TYPE="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    *)
      echo "Unknown option: $1"; print_help; exit 1;;
  esac
done

# Derive annotation/output base paths from eval type
ANNO_BASE="Mountchicken/Rex-Omni-Eval/annotations/${EVAL_TYPE}"
OUTPUT_BASE="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/${EVAL_TYPE}"

# Resolve dataset-specific defaults
case "$DATASET" in
  LVIS)
    DEFAULT_JSONL="$ANNO_BASE/LVIS.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/LVIS"
    ;;
  COCO)
    DEFAULT_JSONL="$ANNO_BASE/COCO.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/COCO"
    ;;
  HierText)
    DEFAULT_JSONL="$ANNO_BASE/HierText.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/HierText"
    ;;
  DocLayNet)
    DEFAULT_JSONL="$ANNO_BASE/DocLayNet.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/DocLayNet"
    ;;
  HumanRef)
    DEFAULT_JSONL="$ANNO_BASE/HumanRef.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/HumanRef"
    ;;
  Dense200)
    DEFAULT_JSONL="$ANNO_BASE/Dense200.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/Dense200"
    ;;
  IC15)
    DEFAULT_JSONL="$ANNO_BASE/IC15.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/IC15"
    ;;
  M6Doc)
    DEFAULT_JSONL="$ANNO_BASE/M6Doc.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/M6Doc"
    ;;
  RefCOCOg_test)
    DEFAULT_JSONL="$ANNO_BASE/RefCOCOg_test.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/RefCOCOg_test"
    ;;
  RefCOCOg_val)
    DEFAULT_JSONL="$ANNO_BASE/RefCOCOg_val.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/RefCOCOg_val"
    ;;
  SROIE)
    DEFAULT_JSONL="$ANNO_BASE/SROIE.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/SROIE"
    ;;
  TotalText)
    DEFAULT_JSONL="$ANNO_BASE/TotalText.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/TotalText"
    ;;
  VisDrone)
    DEFAULT_JSONL="$ANNO_BASE/VisDrone.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/VisDrone"
    ;;
  FSCD_test)
    DEFAULT_JSONL="$ANNO_BASE/FSCD_test.jsonl"
    OUTPUT_DIR="$OUTPUT_BASE/FSCD_test"
    ;;
  *)
    echo "Unsupported dataset: $DATASET"; exit 1;;
esac

# Apply overrides
if [[ -n "$TEST_JSONL" ]]; then
  SELECTED_JSONL="$TEST_JSONL"
else
  SELECTED_JSONL="$DEFAULT_JSONL"
fi

if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Paths
PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
EVAL_JSON="$OUTPUT_DIR/eval_results.json"

# Summary
echo "🚀 Starting Generic Box Evaluation Pipeline"
echo "======================================"
echo "Dataset: $DATASET"
echo "Model  : $MODEL_PATH"
echo "Test   : $SELECTED_JSONL"
echo "Output : $OUTPUT_DIR"

echo "\n📝 Step 1: Inference"
python evaluation/inference_text_prompt.py \
  --model_path "$MODEL_PATH" \
  --test_jsonl_path "$SELECTED_JSONL" \
  --image_root_dir "$IMAGE_ROOT" \
  --save_path "$PRED_JSONL" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --max_new_tokens "$MAX_TOKENS" \
  --min_pixels "$MIN_PIXELS" \
  --max_pixels "$MAX_PIXELS" \
  --question "$QUESTION"

echo "\n📊 Step 2: Metrics (other_metric.py)"
python evaluation/metrics/other_metric.py \
  --data_path "$PRED_JSONL" \
  --output_path "$EVAL_JSON"

echo "\n✅ Done. Results:"
echo "  - Predictions: $PRED_JSONL"
echo "  - Eval JSON  : $EVAL_JSON"
